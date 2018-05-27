import hashlib
import os
import json

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

SENTENCE_START = '<B>'
SENTENCE_END = '<E>'

def read_text_file(text_file):
    lines = []
    with open(text_file, "rt") as f:
        for line in f:
            line = line.decode('utf-8')
            lines.append(line.strip())
    return lines

def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line=="": return line
    if line[-1] in END_TOKENS: return line
    return line + " ."

def get_art_abs(story_file):
    lines = read_text_file(story_file)
    
    # Lowercase everything
#     lines = [line.lower() for line in lines]
    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]
    
    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx,line in enumerate(lines):
        if line == "":
            continue # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = ' '.join(article_lines)

    # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

    return article, abstract

def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()

def dump_out_to_json(hash_path_dict, urls, outpath):
    all_instances = []
    for cur_url in urls:
        cur_hash_code = hashhex(cur_url)
        cur_path = hash_path_dict[cur_hash_code]
        (article, abstract) = get_art_abs(cur_path)
        all_instances.append({'id': cur_hash_code, 'text1': article, 'text2': abstract})
    
    with open(outpath, 'w') as outfile:
        json.dump(all_instances, outfile)
        
def create_json_file(hash_path_dict, urlpath, outpath, batch_size=5000):
    all_urls = read_text_file(urlpath)
    import padding_utils
    batch_spans = padding_utils.make_batches(len(all_urls), batch_size)
    for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
        cur_urls = all_urls[batch_start:batch_end]
        cur_outpath = outpath + ".{}".format(batch_index)
        print("Dump {} instances out to {}".format(len(cur_urls), cur_outpath))
        dump_out_to_json(hash_path_dict, cur_urls, cur_outpath)



def process():
    cnn_in_dir = "/u/zhigwang/zhigwang1/sentence_generation/cnn-dailymail/cnn/cnn/stories"
    daily_mail_in_dir = "/u/zhigwang/zhigwang1/sentence_generation/cnn-dailymail/dailymail/dailymail/stories"
    train_urls = "/u/zhigwang/zhigwang1/sentence_generation/cnn-dailymail/url_lists/all_train.txt"
    val_urls = "/u/zhigwang/zhigwang1/sentence_generation/cnn-dailymail/url_lists/all_val.txt"
    test_urls = "/u/zhigwang/zhigwang1/sentence_generation/cnn-dailymail/url_lists/all_test.txt"
    outdir = "/u/zhigwang/zhigwang1/sentence_generation/cnn-dailymail/data"
    batch_size = 5000
    
    # collect all files
    print('collecting all files')
    in_dirs = [cnn_in_dir, daily_mail_in_dir]
    hash_path_dict = {}
    for in_dir in in_dirs:
        all_paths= os.listdir(in_dir)
        for cur_path in all_paths:
            if not cur_path.endswith(".story"): continue
            cur_hash_code = cur_path[:-len('.story')]
            cur_path = in_dir + "/" + cur_path
            hash_path_dict[cur_hash_code] = cur_path
#             print(cur_path)
#             print(cur_hash_code)
    print('number of files: {}'.format(len(hash_path_dict)))
    print('Creating val.json')
    create_json_file(hash_path_dict, val_urls, outdir + "/val.json", batch_size=batch_size)
    print('Creating test.json')
    create_json_file(hash_path_dict, test_urls, outdir + "/test.json", batch_size=batch_size)
    print('Creating train.json')
    create_json_file(hash_path_dict, train_urls, outdir + "/train.json", batch_size=batch_size)

def generate_tok_commandlines(inpath):
    all_paths = read_text_file(inpath)
    for cur_path in all_paths:
        print("jbsub -q x86_12h -cores 10 -name process_summarization -mem 21G "
        + "/u/zhigwang/workspace/FactoidQA_Java/scripts/process_generation_datasets.sh " 
        + "{} {}.tok 10".format(cur_path, cur_path))
    

if __name__ == "__main__":
    '''
    inpath = "/u/zhigwang/zhigwang1/sentence_generation/cnn/cnn/stories/fffcd65676a501860ae312754e8cefc71f5ddab8.story"
    (article, abstract) = get_art_abs(inpath)
    print(article)
    print(abstract)
    '''
    
#     process()
    #'''
    inpath = "/u/zhigwang/zhigwang1/sentence_generation/cnn-dailymail/data/fof"
#     inpath = "/u/zhigwang/zhigwang1/sentence_generation/mscoco/data/fof"
#     inpath = "/u/zhigwang/zhigwang1/sentence_generation/quora/data/fof"
    generate_tok_commandlines(inpath)
    #'''
#     print("DONE!")