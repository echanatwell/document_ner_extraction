from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
from typing import Dict, Union

from transformers import AutoModelForTokenClassification, LayoutXLMTokenizerFast, LayoutLMv2FeatureExtractor
import numpy as np
import base64
import io
from PIL import Image
import cv2
import torch
from collections import defaultdict
from torch.nn.functional import pad
from ocr_processor import text_recognition
import easyocr
from torchvision import transforms

app = FastAPI()

templates = Jinja2Templates(directory='templates')

model = AutoModelForTokenClassification.from_pretrained('./layoutxlm_ner_extractor') # ("./layoutxlm-finetuned-doc")
tokenizer = LayoutXLMTokenizerFast.from_pretrained('./layoutxlm_ner_extractor') # ("./layoutxlm-base")
feature_extractor = LayoutLMv2FeatureExtractor(ocr_lang="rus") # DEPRECATED => LayoutLMv2ImageProcessor
reader = easyocr.Reader(['ru'])
    
def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

def box_in(box, cluster, threshold=0.9):
    # box=(x0, y0, x1, y1), cluster=(x0, y0, x1, y1)
    S_intersect = max(min(box[2], cluster[2]) - max(box[0], cluster[0]), 0) * max(min(box[3], cluster[3]) - max(box[1], cluster[1]), 0)
    S_box = (box[2] - box[0]) * (box[3] - box[1])

    return (S_intersect / S_box) # >= threshold

def get_clusters(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((9, 9), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=6)

    ctrs, hier = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda x: (cv2.boundingRect(x)[1], -cv2.boundingRect(x)[0]))
    
    clusters = []
    
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        clusters.append((x, y, x+w, y+h))

    return clusters

def img2b64(img):
    """Converts image to base64 encoded string
    Args:
        img: numpy.ndarray
    Returns:
        base64 encoded string
    """
    buffer = io.BytesIO()

    if isinstance(img, np.ndarray):
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        pass

    img.save(buffer, 'PNG')
    buffer.seek(0)

    return base64.b64encode(buffer.getvalue()).decode('utf-8')

@app.get('/api/doc/get-doc-info', response_class=HTMLResponse)
def get_main(request: Request):
    return templates.TemplateResponse(name='main.html', context={'request': request})

@app.post("/api/doc/get-doc-info")
async def extract_context(request: Request):
    # decode image
    print('Start')
    # img = cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(data['img']), dtype=np.uint8),
    #                                 cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) # convert from bytes string to img
    
    form_data = await request.form()
    img_data = form_data['doc-file']
    img_bytes = await img_data.read()
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.cvtColor(cv2.imdecode(arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    
    print('Image decoded')
    
    # preprocess image
    height, width, _ = img.shape
    inputs_words, inputs_boxes = text_recognition(img, reader)

    inputs_boxes = np.array(list(map(normalize_bbox, inputs_boxes, 
                                 [width]*len(inputs_boxes), [height]*len(inputs_boxes))))

    encoding = tokenizer(inputs_words, boxes=inputs_boxes, return_offsets_mapping=True, 
                         return_tensors="pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Preprocessed')
    model.to(device)
    
    # split long text
    input_len = encoding.input_ids.shape[1]

    encoding.input_ids = pad(encoding.input_ids, 
                             pad=(0, 512 - (input_len % 512)), value=1.).reshape(-1, 512)
    encoding.attention_mask = pad(encoding.attention_mask, 
                                  pad=(0, 512 - (input_len % 512)), value=0.).reshape(-1, 512)
    encoding.bbox = pad(encoding.bbox, 
                                  pad=(0, 0, 0, 512 - (input_len % 512)), value=0.).reshape(-1, 512, 4)
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((224, 224))])
    
    img_tensor = transform(img)
    normalized_image_tensor = torch.unsqueeze(img_tensor, 0)
    print('Splitted')
    
    # get predictions
    with torch.no_grad():
        outputs = model(input_ids=encoding.input_ids.to(device),
                        attention_mask=encoding.attention_mask.to(device),
                        bbox=encoding.bbox.to(device),
                        image=normalized_image_tensor.to(device),
                       )
    predictions = outputs.logits.argmax(-1).reshape(-1).tolist()
    token_boxes = encoding.bbox.reshape(-1, 4).tolist()
    
    height, width, _ = img.shape
    
    is_subword = np.array(encoding.offset_mapping.tolist())[:,:,0].reshape(-1) != 0

    true_predictions = [model.config.id2label[pred] for idx, pred in enumerate(predictions[:len(is_subword)]) if not is_subword[idx]]
    true_boxes = [tuple(unnormalize_box(box, width, height)) for idx, box in enumerate(token_boxes[:len(is_subword)]) if not is_subword[idx]]
    preds_boxes = [(box, pred) for i, (pred, box) in enumerate(zip(true_predictions, true_boxes)) if box not in true_boxes[:i]]
    words = tokenizer.decode(encoding['input_ids'][0]).split(' ')

    # get_clusters
    clusters = get_clusters(img)

    entities_by_cls = defaultdict(list)
    for (box, tag), word in zip(preds_boxes[1:], inputs_words):
        # print(box)
        # print([int(box_in(box, clusters[i])) for i in range(len(clusters))])
        cluster = int(np.argmax([int(box_in(box, clusters[i])) for i in range(len(clusters))]))
        entities_by_cls[cluster].append((word, tag))
        
    # extract ners
    ners_dict = defaultdict(list)
    entity = []
    for cluster in sorted(entities_by_cls.keys()):
        for word, tag in entities_by_cls[cluster]:
            if tag[:2] == 'B-':
                if len(entity) > 0:
                    only_ent = [ent for ent in entity if ent[1] != 'O']
                    ners_dict[only_ent[0][1][2:]].append(' '.join([e[0] for e in only_ent]))
                    entity = []
            entity.append((word, tag))
    
    if len(entity) > 0:
        only_ent = [ent for ent in entity if ent[1] != 'O']
        ners_dict[only_ent[0][1][2:]].append(' '.join([e[0] for e in only_ent]))
    
    img_ = img.copy()
    for (box, pred) in [(box, pred) for (box, pred) in preds_boxes[1:] if pred != 'O']:
        start = (int(box[0]), int(box[1]))
        end = (int(box[2]), int(box[3]))
        cv2.rectangle(img_, start, end, (255, 0, 0), 1)
        cv2.putText(img_, pred, start, cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255), 1, cv2.LINE_AA)


    # ners_dict['JT'] = sorted(ners_dict['JT'], key=lambda x: x[2])
    return templates.TemplateResponse(name='report.html', context={
            'request': request,
            'senderOrganization': str(ners_dict['ORG']) if ners_dict['ORG'] else 'Не найдено',
            'senderNumber': str(ners_dict['MN']) if ners_dict['MN'] else 'Не найдено',
            'senderDate': str(ners_dict['DS'][0]) if ners_dict['DS'] else 'Не найдено',
            'addresseeName': str(ners_dict['ADR']) if ners_dict['ADR'] else 'Не найдено',
            'addresseeJobTitle': str(ners_dict['JT'][:3]) if ners_dict['JT'] else 'Не найдено',
            'senderName': str(ners_dict['SND']) if ners_dict['SND'] else 'Не найдено',
            'senderJobTitle': str(ners_dict['JT'][-3:]) if ners_dict['JT'] else 'Не найдено',
            'executorName': str(ners_dict['EXR']) if ners_dict['EXR'] else 'Не найдено',
            'executorPhone': str(ners_dict['PHN']) if ners_dict['PHN'] else 'Не найдено', 
            'executorEmail': str(ners_dict['MAIL']) if ners_dict['MAIL'] else 'Не найдено',
            'location': str(ners_dict['LOC']) if ners_dict['LOC'] else 'Не найдено',
            'inn': str(ners_dict['INN']) if ners_dict['INN'] else 'Не найдено',
            'kpp': str(ners_dict['KPP']) if ners_dict['KPP'] else 'Не найдено',
            'image': img2b64(img_)
           })


if __name__ == '__main__':
    uvicorn.run('main:app', port=8000, host='0.0.0.0', log_level='info', reload=False)
