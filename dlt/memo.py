import json

# JSON 파일 읽기
with open("dlt/dataset/train_canva.json", 'r') as file:
    data = json.load(file)

# 모든 'type'을 저장할 리스트
types = []

# 모든 presentations와 slides를 순회
for presentation in data['presentations']:
    for slide in presentation['slides']:
        for content in slide['contents']:
            # 'type' 추출 및 저장
            content_type = content['type']
            types.append(content_type)

# 중복 제거를 위해 set으로 변환 후 다시 리스트로
unique_types = list(set(types))

print("Unique types:", unique_types)
