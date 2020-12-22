import Algorithmia
import json
import string
import argparse

def parse(args):
    raw_data = open(args.dataset, 'r')
    f = open(args.output_path, 'w')
    #algorithmia client
    client = Algorithmia.client('simx9rqQpbdFH7H8CaJ+DFHNGOZ1')
    algo = client.algo('allenai/dependency_parsing/0.1.0')

    for l in raw_data:
        res_json = {}
        line_json = json.loads(l)
        sentence = line_json['sentence']
        res_json['sentence'] = sentence
        
        #get dependency parse after filling in sent 1
        sentence_1 = sentence.replace('_', line_json['option1'])
        inp = {"sentence": sentence_1}
        allenai_parse = algo.pipe(inp).result
        parse = translate(allenai_parse)
        res_json['sentence_1'] = parse

        #get parse after filling in sent 2
        sentence_2 = sentence.replace('_', line_json['option2'])
        inp = {"sentence": sentence_2}
        allenai_parse = algo.pipe(inp).result
        parse = translate(allenai_parse)
        res_json['sentence_2'] = parse
        res_json['answer'] = line_json['answer']
        
        f.write(json.dumps(res_json))
        f.write('\n')

def translate(allenai_parse):
    heads = allenai_parse['predicted_heads']
    sentence = allenai_parse['words']
    
    head_words = list()
    for i in range(len(heads)):
        head_words.append((heads[i], sentence[i]))
    
    head_words.sort()
    result_sent = list()
    for head, word in head_words:
        result_sent.append(word)
    return ' '.join(result_sent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parse WinoGrande dataset into dependency parsed sentences')
    
    parser.add_argument('--dataset', type=str, help='path to WinoGrande dataset in jsonl format', 
            default='winogrande_1.1/train_s.jsonl')
    parser.add_argument('--output_path', type=str, help='path for output file', default='parsed_data_s.jsonl')
    
    args = parser.parse_args()
    parse(args)