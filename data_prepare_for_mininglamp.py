import os
import sys
from io import open
import csv
import random

# read_txt and csv data
def read_and_write(file_index,data_path,label_path,output_path,sub_fix= "",_type="train"):

    results = []
    all_labels = set()

    for index in file_index:
        with open(os.path.join(data_path,"{}.txt".format(index)),"r",encoding="utf8") as f,open(os.path.join(label_path,"{}.csv".format(index)),"r",encoding="utf8") as g:
            sentence = f.read().strip()
            tags = ["O"] * len(sentence)
            labels = csv.reader(g)
            
            for label_i,(_,class_name,pos_b,pos_e,entity) in enumerate(labels):

                if label_i == 0:
                    continue
                if int(pos_e) <= len(sentence):

                    if int(pos_e) > int(pos_b):
                        tags[int(pos_b):int(pos_e)+1] = ["B-{}".format(class_name).upper()] + ["I-{}".format(class_name).upper()] * (int(pos_e)-int(pos_b))
                        all_labels.add("B-{}".format(class_name).upper())
                        all_labels.add("I-{}".format(class_name).upper())
                    elif int(pos_e) == int(pos_b):
                        tags[int(pos_b)] =  "S-{}".format(class_name).upper()
                        all_labels.add("S-{}".format(class_name).upper())
                    else:
                        continue

            if  len(tags) != len(sentence):
                print("train index is ",index)
                print(f"len tag is {len(tags)},len sentence is {len(sentence)}")
                print(f"tags:{tags},sentences:{sentence}")
                continue

            for word,tag in zip(sentence,tags):
                results.append("{} {}\n".format(word,tag))
            
            results.append(("\n"))
    print("loading train data complete, start write data")
    
    with open(os.path.join(output_path,"{}.data".format(_type)+sub_fix ),"w",encoding="utf8") as wirter:
        

        wirter.writelines(results)

    if sub_fix == "all":
        with open(os.path.join(output_path,"label.txt"+sub_fix),"w",encoding="utf8") as wirter:
            wirter.write("\n".join(list(all_labels)))

    print("write train data complete!!!")

    return all_labels







def get_ner_data_bios(data_path,_type="train",output_path=None,sample_dev=False,seed=42,dev_size=0.25):

    if output_path is None:
            output_path = data_path
        
    else:
        if not os.path.exists(ouput_path):
            os.makedirs(output_path)

    if _type == "train" and not sample_dev:

        train_path = os.path.join(data_path,"train")
        train_data_path = os.path.join(train_path,"data")
        train_label_path = os.path.join(train_path,"label")

        
        train_file_len_data = len([i for i in os.listdir(train_data_path) if ".txt" in i])
        train_file_len_label = len([i for i in os.listdir(train_label_path) if ".csv" in i ])
        assert train_file_len_label == train_file_len_data
        trian_file_indexs = range(len(train_file_len_data))

        read_and_write(trian_file_indexs,train_data_path,train_label_path,output_path,sub_fix="all")

        

    elif _type == "text":
        results = []
        

        test_data_path = os.path.join(data_path,"test")
        for test_file in [i for i in os.listdir(test_data_path) if ".txt" in i]:
            results.append(open(os.path.join(test_data_path,test_file)).read().strip() +"\n")
        
        with open(os.path.join(output_path,"test.data"),"w",encoding="utf8") as wirter:
            wirter.writelines(results)

        print("write  test data complete!!!")

    else:
        results = []
        all_labels = set()
        train_path = os.path.join(data_path,"train")
        train_data_path = os.path.join(train_path,"data")
        train_label_path = os.path.join(train_path,"label")

        

        
        train_file_len_data = len([i for i in os.listdir(train_data_path) if ".txt" in i])
        train_file_len_label = len([i for i in os.listdir(train_label_path) if ".csv" in i ])
        assert train_file_len_label == train_file_len_data
        
        # 产生dev数据
        random.seed(seed)
        all_file_index = list(range(train_file_len_data))
        random.shuffle(all_file_index)
        train_file_indexs = all_file_index[:int(len(all_file_index)*(1-dev_size))]
        dev_file_indexs = all_file_index[int(len(all_file_index)*(1-dev_size)):]

        train_labels = read_and_write(train_file_indexs,train_data_path,train_label_path,output_path)
        dev_labels = read_and_write(dev_file_indexs,train_data_path,train_label_path,output_path,_type="dev")
        all_labels = train_labels | dev_labels
        with open(os.path.join(output_path,"label.txt"),"w",encoding="utf8") as wirter:
            wirter.write("\n".join(list(all_labels)))
        print("all done！！！")

        


        
        



if __name__ == "__main__":
    # get_ner_data_bios("./data/mininglamp")
    get_ner_data_bios("./data/mininglamp",_type="train",sample_dev=True)







        
            
        
                        

                
                



            


