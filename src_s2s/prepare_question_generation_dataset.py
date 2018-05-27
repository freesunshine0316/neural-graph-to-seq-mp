import json

def read_all_SQuAD_questions(inpath):
    with open(inpath) as dataset_file:
        dataset_json = json.load(dataset_file, encoding='utf-8')
        dataset = dataset_json['data']
    all_questions = []
    for article in dataset:
#         title = article['title']
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for question in paragraph['qas']:
                question_text = question['question']
                question_id = question['id']
                answers = question['answers']
    return all_questions

