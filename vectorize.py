import jieba
import re

ENCODER_F = "./materials/yellowchick.question"
DECODER_F = "./materials/yellowchick.answer"
DICT_F = "./materials/word_dict.txt"
MAX_VOCAB_NUM = 30000
ENCODER_VOCAB = './preprocessing/enc.vocabulary'
ENCODER_SEG = './preprocessing/enc.segement'
DECODER_VOCAB = './preprocessing/dec.vocabulary'
DECODER_SEG = './preprocessing/dec.segement'
ENCODER_VEC = "./preprocessing/enc.vector"
DECODER_VEC = "./preprocessing/dec.vector"

class preprocessing():
    __PAD__ = 0
    __GO__ = 1
    __EOS__ = 2
    __UNK__ = 3
    vocab = ['__PAD__', '__GO__', '__EOS__', '__UNK__']
    def __init__(self):
        self.encoderFile = ENCODER_F
        self.decoderFile = DECODER_F
        self.dictFile = DICT_F
        jieba.load_userdict(self.dictFile)

    def wordToVocabulary(self, originFile, vocabFile, segementFile):

        vocabulary = {}
        with open(segementFile, "w") as sege:
            with open(originFile, 'r') as en:
                count = 0
                for sent in en.readlines():
                    count += 1
                    # 去标点    
                    words = jieba.lcut(sent.strip())

                    for word in words:
                        if word in vocabulary:  
                            vocabulary[word] += 1  
                        else:  
                            vocabulary[word] = 1  
                        sege.write(word+" ")
                    sege.write("\n")
                    if count % 10000 == 0:
                        print("progresss:", count)
        print("total length",len(vocabulary))

        vocabulary_list = self.vocab + sorted(vocabulary, key=vocabulary.get, reverse=True)  

        if len(vocabulary_list) > MAX_VOCAB_NUM:  
           vocabulary_list = vocabulary_list[:MAX_VOCAB_NUM]  
        print(" 词汇表大小:", len(vocabulary_list))  
        with open(vocabFile, "w") as ff:  
            for word in vocabulary_list:  
                ff.write(word + "\n")  

    def toVec(self, segementFile, vocabFile, vecFile):
        word_dicts = {}
        vec = []
        with open(vocabFile, "r") as dict_f:
            for index, word in enumerate(dict_f.readlines()):
                word_dicts[word.strip()] = index

        with open(vecFile, "w") as f:
            with open(segementFile, "r") as sege_f:
                for sent in sege_f.readlines():
                    sents = [i.strip() for i in sent.split(" ")[:-1]]
                    vec.extend(sents)
                    for word in sents:
                        f.write(str(word_dicts.get(word, self.__UNK__))+" ")
                    f.write("\n")

            

    def main(self):
        # 获得字典
        print("encoder vocabulary preparing")
        self.wordToVocabulary(self.encoderFile, ENCODER_VOCAB, ENCODER_SEG)
        print("decoder vocabulary preparing")
        self.wordToVocabulary(self.decoderFile, DECODER_VOCAB, DECODER_SEG)
        # 转向量
        self.toVec(ENCODER_SEG, ENCODER_VOCAB, ENCODER_VEC)
        self.toVec(DECODER_SEG, DECODER_VOCAB, DECODER_VEC)

        print('finish preprocessing')

if __name__ == '__main__':
    pre = preprocessing()
    pre.main()
