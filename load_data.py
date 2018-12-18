# -*- coding: utf-8 -*

"""
some texts in description is list(has more than 1 description such as 补充说明 ; approach: combine string
future: normalize?
"""
import os
import json
#/home/data/healthData/healthData/files/data/120ask/extracted/


def printLabel(labels):
    """
    print list of jibing tags
    :param label: list of jibing tags
    :return: none
    """
    for label in labels:
        print label

def inspect(f):
    # for i, label in enumerate(labels):
    #     for item in label:
    #         print '{} {} {}'.format(i, item.encode('utf-8'), len(label))

    # check set of label length
    # [1, 2, 3, 4, 5, 6, 7, 8, ... 40, 42, 45, 46, 47, 51, 52, 53]
    # no missing labels : good sign; but 53 is ridiculous
    if f == 2:
        temp = list()
        for i, label in enumerate(labels):
            temp.append(len(label))

            # check data with 53 labels
            if len(label) == 20:
                print texts[i]
                printLabel(label)

        label_length_set = list(set(temp))
        print label_length_set

    # print json line of 40 jibing tags
    if f == 3:
        if len(single_data_object["jibing_tags"]) == 40:
            print line

    # print length of jibing tags distribution
    if f ==4:
        tags_length_distribution[len(single_data_object["jibing_tags"])] += 1


def count_tag_leakage_for_one_label_data(txts,lbls):
    count_lbl_eq_1 = 0
    count_tag_leak_eq_1 = 0
    stop = 0

    for i, lbl in enumerate(lbls):
        if len(lbl) == 1:
            count_lbl_eq_1 += 1

            if lbl[0] in txts[i]:
                count_tag_leak_eq_1 += 1

            # print several examples
            if 80< stop < 90:
                a = lbl[0]
                b = txts[i]
                print b
                print a
            stop+=1

    ratio = count_tag_leak_eq_1/count_lbl_eq_1

    print 'label_eq_1 {} leak {} -> ratio label leakage of 1 class{}'.format(count_lbl_eq_1,count_tag_leak_eq_1,ratio)

def print_taglength_distribution():
    tags_length_distribution = {i: 0 for i in range(60)}
    print tags_length_distribution

def print_output_distribution():
    """
    observe distribution of output tags: assume distribution at least follow long-tails distribution
    :return:
    """
    raise NotImplementedError


def load_data(datapath,filename,keep_only_single_output=False):
    if os.path.exists(os.path.join(datapath,filename)):
        print('current data path: {} '.format(os.path.abspath(os.path.join(datapath,filename))))
        with open(os.path.join(datapath, filename)) as f:

            texts = []
            labels = []

            temp = []

            for line in f.readlines():
                # turn json string into object
                single_data_object = json.loads(line)

                if keep_only_single_output:
                    if len(single_data_object["jibing_tags"]) == 1:

                        if type(single_data_object["Question"]["description"]) == list:
                            texts.append('/'.join(single_data_object["Question"]["description"]))
                        else:
                            # read text data
                            texts.append(single_data_object["Question"]["description"])

                        # read jibing labels : list of jibings
                        labels.append(single_data_object["jibing_tags"])

                else:
                    # still need to further add (e.g. deal with input list)
                    texts.append(single_data_object["Question"]["description"])
                    labels.append(single_data_object["jibing_tags"])


                # inspect(3)
                # inspect(4)


            # temporary code for observe data
            # print_taglength_distribution()
            count_tag_leakage_for_one_label_data(texts,labels)

            print 'n of texts : {} '.format(len(texts))
            print 'n of labels : {} '.format(len(labels))

        return texts, labels


if __name__ == '__main__':
    DATAPATH = '/home/data/healthData/healthData/files/data/120ask/extracted/'
    DATAFILE = 'qa_trainset.json'
    texts, labels = load_data(datapath=DATAPATH, filename=DATAFILE)




