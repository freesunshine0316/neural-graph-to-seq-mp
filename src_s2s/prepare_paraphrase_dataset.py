import json
import random
import re

def read_MSCOCO(inpath):
    # load json file
    with open(inpath) as dataset_file:
        dataset_json = json.load(dataset_file, encoding='utf-8')
        annotations = dataset_json['annotations']
    print(len(annotations))
    
    # dispatch each caption to its corresponding image
    image_captions_dict = {}
    for annotation in annotations:
        image_id = annotation['image_id']
        id = annotation['id']
        caption = annotation['caption']
        captions = None
        if image_captions_dict.has_key(image_id):
            captions = image_captions_dict[image_id]
        else:
            captions = []
        captions.append(caption)
        image_captions_dict[image_id] = captions
    
    # check number of captions for each image
    all_instances = []
    for image_id in image_captions_dict.keys():
        captions = image_captions_dict[image_id]
        random.shuffle(captions)
#         print(len(captions))
        all_instances.append((image_id, captions))
    return all_instances

def read_Quora(inpath):
    with open(inpath, "rt") as f:
        for line in f:
            line = line.decode('utf-8')
            line = line.strip()
            items = re.split('\t', line)


def dump_out_to_json(instances, outpath):
    json_instances = []
    for (image_id, captions) in instances:
        json_instances.append({'id': str(image_id), 'text1': captions[0], 'text2': captions[1]})
#         json_instances.append({'id': str(image_id) + "-2", 'text1': captions[2], 'text2': captions[3]})
    with open(outpath, 'w') as outfile:
        json.dump(json_instances, outfile)
        
def create_json_file(all_instances, outpath, batch_size=5000):
    import padding_utils
    batch_spans = padding_utils.make_batches(len(all_instances), batch_size)
    for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
        cur_instances = all_instances[batch_start:batch_end]
        cur_outpath = outpath + ".{}".format(batch_index)
        print("Dump {} instances out to {}".format(len(cur_instances), cur_outpath))
        dump_out_to_json(cur_instances, cur_outpath)

if __name__ == "__main__":
    ''' # create mscoco dataset
    dataset = "train"
    inpath = "/u/zhigwang/zhigwang1/sentence_generation/mscoco/annotations/captions_" + dataset + "2014.json"
    outpath = "/u/zhigwang/zhigwang1/sentence_generation/mscoco/data/" + dataset + ".json"
    all_instances = read_MSCOCO(inpath)
    batch_size = 5000
    create_json_file(all_instances, outpath, batch_size=batch_size)
    '''
    
    # create quora dataset
    rawpath = "/u/zhigwang/zhigwang1/sentence_match/quora/quora_duplicate_questions.tsv"
    batch_size = 10000
    # load all pairs
    print('Loading all question pairs ...')
    id_instances_dict = {}
    with open(rawpath, "rt") as f:
        for line in f:
            line = line.decode('utf-8').strip()
            if not line.endswith("1"): continue
            items = re.split('\t', line)
            cur_id = items[0]
            sent1 = items[3]
            sent2 = items[4]
            id_instances_dict[cur_id] = (sent1, sent2)
    print(len(id_instances_dict))
    
    for dataset in ['dev', 'test', 'train']:
        # collect all isntances
        inpath = "/u/zhigwang/zhigwang1/sentence_match/quora/" + dataset + ".tsv"
        outpath = "/u/zhigwang/zhigwang1/sentence_generation/quora/data/" + dataset + ".json"
        print(inpath)
        all_instances = []
        with open(inpath, "rt") as f:
            for line in f:
                line = line.decode('utf-8').strip()
                if not line.startswith("1"): continue
                items = re.split('\t', line)
                cur_id = items[3]
                all_instances.append((cur_id, id_instances_dict[cur_id]))
        create_json_file(all_instances, outpath, batch_size=batch_size)
     


    print('DONE!')