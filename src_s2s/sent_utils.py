import re

class QASentence(object):
    def __init__(self, rawText, annotation, ID_num=None, isLower=False, end_sym=None):
        self.rawText = rawText
        self.annotation = annotation
        self.tokText = annotation['toks']
        # it's the answer sequence
        if end_sym != None:
            self.rawText += ' ' + end_sym
            self.tokText += ' ' + end_sym
        if isLower: self.tokText = self.tokText.lower()
        self.words = re.split("\\s+", self.tokText)
        self.startPositions = []
        self.endPositions = []
        positions = re.split("\\s+", annotation['positions'])
        for i in xrange(len(positions)):
            tmps = re.split("-", positions[i])
            self.startPositions.append(int(tmps[1]))
            self.endPositions.append(int(tmps[2]))
        self.POSs = annotation['POSs']
        self.NERs = annotation['NERs']
        if annotation.has_key('spans'): self.syntaxSpans = annotation['spans']
        self.length = len(self.words)
        self.ID_num = ID_num

        self.index_convered = False
        self.chunk_starts = None

    def chunk(self, maxlen):
        self.words = self.words[:maxlen]
        self.startPositions = self.startPositions[:maxlen]
        self.endPositions = self.endPositions[:maxlen]
        self.POSs = self.POSs[:maxlen]
        self.NERs = self.NERs[:maxlen]

        if self.index_convered:
            self.word_idx_seq = self.word_idx_seq[:maxlen]
            self.char_idx_seq = self.char_idx_seq[:maxlen]
            self.POS_idx_seq = self.POS_idx_seq[:maxlen]
            self.NER_idx_seq = self.NER_idx_seq[:maxlen]

        self.length = len(self.words)

    def TokSpan2RawSpan(self, startTokID, endTokID):
        start = self.startPositions[startTokID]
        end = self.endPositions[endTokID]
        return (start, end)

    def RawSpan2TokSpan(self, start, end):
        startTokID = -1
        endTokID = -1
        for i in xrange(len(self.startPositions)):
            if self.startPositions[i] == start:
                startTokID = i
            if self.endPositions[i] == end:
                endTokID = i
        return (startTokID, endTokID)

    def getRawChunk(self, start, end):
        if end>len(self.rawText):
            return None
        return self.rawText[start:end]

    def getRawChunkWithTokSpan(self, startTokID, endTokID):
        start = self.startPositions[startTokID]
        end = self.endPositions[endTokID]
        return self.rawText[start:end]

    def getTokChunk(self, startTokID, endTokID):
        curWords = []
        for i in xrange(startTokID,endTokID+1):
            curWords.append(self.words[i])
        return " ".join(curWords)

    def get_length(self):
        return self.length

    def get_max_word_len(self):
        max_word_len = 0
        for word in self.words:
            cur_len = len(word)
            if max_word_len < cur_len: max_word_len = cur_len
        return max_word_len

    def get_char_len(self):
        char_lens = []
        for word in self.words:
            cur_len = len(word)
            char_lens.append(cur_len)
        return char_lens

    def convert2index(self, word_vocab, char_vocab, POS_vocab, NER_vocab, max_char_per_word=-1):
        if self.index_convered: return # for each sentence, only conver once

        if word_vocab is not None:
            self.word_idx_seq = word_vocab.to_index_sequence(self.tokText)

        if char_vocab is not None:
            self.char_idx_seq = char_vocab.to_character_matrix(self.tokText, max_char_per_word=max_char_per_word)

        if POS_vocab is not None:
            self.POS_idx_seq = POS_vocab.to_index_sequence(self.POSs)

        if NER_vocab is not None:
            self.NER_idx_seq = NER_vocab.to_index_sequence(self.NERs)

        self.index_convered = True

    def collect_all_possible_chunks(self, max_chunk_len):
        if self.chunk_starts is None:
            self.chunk_starts = []
            self.chunk_ends = []
            for i in xrange(self.length):
                cur_word = self.words[i]
                if cur_word in ".!?;": continue
                for j in xrange(i, i+max_chunk_len):
                    if j>=self.length: break
                    cur_word = self.words[j]
                    if cur_word in ".!?;": break
                    self.chunk_starts.append(i)
                    self.chunk_ends.append(j)
        return (self.chunk_starts, self.chunk_ends)

    def collect_all_entities(self):
        items = re.split("\\s+", self.NERs)
        prev_label = "O"
        cur_start = -1
        chunk_starts = []
        chunk_ends = []
        for i in xrange(len(items)):
            cur_label = items[i]
            if cur_label != prev_label:
                if cur_start != -1:
                    chunk_starts.append(cur_start)
                    chunk_ends.append(i-1)
                    cur_start = -1
                if cur_label != "O":
                    cur_start = i
            prev_label = cur_label
        if cur_start !=-1:
            chunk_starts.append(cur_start)
            chunk_ends.append(len(items)-1)
        return (chunk_starts, chunk_ends)

    def collect_all_syntax_chunks(self, max_chunk_len):
        if self.chunk_starts is None:
            self.chunk_starts = []
            self.chunk_ends = []
            self.chunk_labels = []
            all_spans = re.split("\\s+", self.syntaxSpans)
            for i in xrange(len(all_spans)):
                cur_span = all_spans[i]
                items = re.split("-", cur_span)
                cur_start = int(items[0])
                cur_end = int(items[1])
                cur_label = items[2]
                if cur_end-cur_start>=max_chunk_len: continue
                self.chunk_starts.append(cur_start)
                self.chunk_ends.append(cur_end)
                self.chunk_labels.append(cur_label)
        return (self.chunk_starts, self.chunk_ends, self.chunk_labels)

if __name__ == "__main__":
    import NP2P_data_stream
    inpath = "/u/zhigwang/zhigwang1/sentence_generation/cnn-dailymail/data/val.json.tok"
    all_instances,_ = NP2P_data_stream.read_all_GenerationDatasets(inpath, isLower=True)
    sample_instance = all_instances[0][1]
    print('Raw text: {}'.format(sample_instance.rawText))
    (chunk_starts, chunk_ends, chunk_labels) = sample_instance.collect_all_syntax_chunks(5)
    for i in xrange(len(chunk_starts)):
        cur_start = chunk_starts[i]
        cur_end = chunk_ends[i]
        cur_label = chunk_labels[i]
        cur_text = sample_instance.getTokChunk(cur_start, cur_end)
        print("{}-{}-{}:{}".format(cur_start, cur_end, cur_label, cur_text))
    print("DONE!")
