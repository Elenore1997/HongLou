import re
import jieba
import jieba.posseg as pseg
from pyecharts import options as opts
from pyecharts.charts import WordCloud, Bar
from pyecharts.globals import SymbolType
import json
import codecs
import pandas as pd
from collections import OrderedDict
from py2neo import Graph,Node,Relationship
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
from sklearn.decomposition import PCA
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


def cut_chapter():
    """
    切分章节
    原本小说是所有章节在一个txt内，现在通过正则，
    按章节进行存储
    """
    with open('fiction_pre.txt', encoding='utf-8')as r:
        lines = r.readlines()
    pattern = re.compile('第[一二三四五六七八九十百零]+回 ')

    chapter_content = ''
    chapter_num = 0
    for line in lines:
        if re.findall(pattern, line):
            with open('chapters_pre/' + str(chapter_num) + '.txt', 'a+', encoding='utf-8')as w:
                w.write(chapter_content + '\n')
            chapter_content = line
            chapter_num += 1
        else:
            chapter_content += line


def character_wordcloud():
    """
    人物词云（频数）
    根据小说中人物登场次数来制作词云
    :return:
    """
    jieba.load_userdict("character.txt")  # 加载人物 nr
    with open('fiction.txt', encoding='utf-8')as r:
        text = r.read()
    p = pseg.cut(text)  # return: generator

    character_list = []
    with open('character.txt', encoding='utf-8')as r:
        lines = r.readlines()
    for line in lines:
        name = line.split(' ')[0]
        character_list.append(name)
    fre_char_dist = {}

    for word, tag in p:
        if tag == 'nr' and word in character_list:
            if word in fre_char_dist.keys():
                fre_char_dist[word] += 1
            else:
                fre_char_dist[word] = 0

    fre_char_list = list(fre_char_dist.items())

    def wordcloud_diamond() -> WordCloud:
        c = (
            WordCloud()
            .add("", fre_char_list, word_size_range=[20, 100], shape=SymbolType.DIAMOND)
            .set_global_opts(title_opts=opts.TitleOpts(title="Characters WordCloud"))
        )
        return c
    # 生成html，进行可视化
    wordcloud_diamond().render('wordcloud.html')


def relation_count():
    """
    统计每种关系的次数
    数据来源ref：https://github.com/DesertsX/gulius-projects/blob/master/3_InteractiveGraph_HongLouMeng/InteractiveGraph_HongLouMeng.json
    :return:
    """
    with codecs.open('relative.json', 'r', encoding='utf-8') as json_str:
        json_dict = json.load(json_str)
    edges = json_dict['data']['edges']
    edges_df = pd.DataFrame(edges)
    edges_rela = edges_df.label.value_counts()
    # 所有关系
    relations = edges_df.label
    relations = relations.tolist()
    relations = list(set(relations))
    # 每种关系对应的频数
    rela_counts = []
    for each in relations:
        rela_counts.append(int(edges_rela[each]))
    # 柱状图
    bar = Bar()
    bar.add_xaxis(relations)
    bar.add_yaxis('频次', rela_counts)
    bar.set_global_opts(title_opts=opts.TitleOpts(title="红楼梦关系频次表"))
    bar.render('relation_count.html')


def most_appear_per_chapter():
    """
    每一回出场次数最多的角色
    用柱状图进行可视化
    :return:
    """
    jieba.load_userdict("character.txt")  # 加载人物 nr
    character_list = []
    with open('character.txt', encoding='utf-8')as r:
        lines = r.readlines()
    for line in lines:
        name = line.split(',')[0]
        character_list.append(name)

    counts_per_chapter = []
    char_per_chapter = []

    for i in range(120):
        with open('chapters/' + str(i+1) + '.txt', encoding='utf-8')as r:
            text = r.read()
        p = pseg.cut(text)  # return: generator

        fre_char_dist = OrderedDict()
        for word, tag in p:
            if tag == 'nr' and word in character_list:
                if word in fre_char_dist.keys():
                    fre_char_dist[word] += 1
                else:
                    fre_char_dist[word] = 1

        key = list(fre_char_dist.keys())[0]
        counts_per_chapter.append(fre_char_dist[key])
        char_per_chapter.append(key + '-第' + str(i+1) + '回')
        fre_char_dist.clear()
    # print(len(counts_per_chapter))
    # print(len(char_per_chapter))
    bar = Bar()
    bar.add_xaxis(char_per_chapter)
    bar.add_yaxis('频次', counts_per_chapter)
    bar.set_global_opts(title_opts=opts.TitleOpts(title="红楼梦每一回出场次数最多的人物"),
                        toolbox_opts=opts.ToolboxOpts(),
                        datazoom_opts=[opts.DataZoomOpts()])
    bar.render('most_appear_per_chapter.html')


def cut_sent(para):  # 中文断句
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    return para.split("\n")


def co_occur_graph():
    """
    根据共现关系画出图谱
    以每两句作为一个窗口，识别窗口内的角色
    将这些角色bi-gram进行分组，创建节点，边则为二者共现次数
    边颜色越深，共现越频繁，证明人物关联越密切
    :return:
    """
    edge_dict = {}
    jieba.load_userdict("character.txt")  # 加载人物 nr
    with open('fiction_pre.txt', encoding='utf-8')as r:
        lines = r.readlines()
    text = "".join(line.strip() for line in lines)
    sents_list = cut_sent(text)
    # print(sents_list)
    # print(len(sents_list))
    window_sents = []
    window_size = 2  # 窗口大小2，即统计每两句话中出现的角色
    for i in range(0, len(sents_list), window_size):
        window_sents.append("".join(sents_list[i: i + window_size]))
    # print(window_sents)
    # print(len(window_sents))
    character_list = []
    with open('character.txt', encoding='utf-8')as r:
        ch_lines = r.readlines()
    for line in ch_lines:
        name = line.split(' ')[0]
        character_list.append(name)

    co_occur_name = []
    for line in window_sents:
        poss = pseg.cut(line)
        for word, tag in poss:
            if tag == 'nr' and word in character_list and word != '雨村':  # 雨村-门子 两个经常共现，但是与其他人联系不多，
                co_occur_name.append(word)                                # 在图中偏离，为了可视化的效果便删除了
        bi_gram = nltk.ngrams(co_occur_name, 2)
        for each in bi_gram:
            # print(each)
            # print(type(each))
            if each[0] == each[1]:
                continue
            else:
                if each in edge_dict.keys():
                    edge_dict[each] += 1
                elif (each[1], each[0]) in edge_dict.keys():
                    edge_dict[(each[1], each[0])] += 1
                else:
                    edge_dict[each] = 1
        co_occur_name = []

    # sum_values = sum(edge_dict.values())

    G = nx.Graph()
    thresh = 10  # 设置阈值，共现超过10次才显示
    for key in edge_dict.keys():
        if edge_dict[key] >= thresh:
            G.add_edge(key[0], key[1], weight=edge_dict[key])
    # print(G.number_of_nodes())
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    pos = nx.spring_layout(G)
    nx.draw(G, pos, edge_color=weights, edgelist=edges, width=1.0, edge_cmap=plt.cm.Blues, with_labels=True,
            node_size=220, node_color=range(G.number_of_nodes()), cmap=plt.cm.Blues, font_size=4)
    plt.savefig('co_occur_new.jpg', dpi=300)
    plt.show()




def relation_graph():
    """
    用neo4j可视化人物关系图谱
    节点为角色
    边为人物关系
    :return:
    """
    test_graph = Graph(
        "http://localhost:7474",
        username="neo4j",
        password="sincejune1997."
    )
    test_graph.delete_all()

    # 人物关系图谱
    with codecs.open('relative.json', 'r',encoding='utf-8') as json_str:
        json_dict = json.load(json_str)
    nodes = json_dict['data']['nodes']
    edges = json_dict['data']['edges']
    person_nodes = []
    for num, node in enumerate(nodes):
        if node['categories'][0] == 'person':
            # person = node['label']
            person_nodes.append(node)
    person_df = pd.DataFrame(person_nodes)
    # print(person_df)
    char_list = list(set(person_df.label))  # 角色列表
    # print(char_list)
    edges_df = pd.DataFrame(edges)
    # print(edges_df.head())
    relation_list = list(set(edges_df.label))  # 关系列表

    def relation2id(relation):  # 根据关系，得到该关系的两个节点id
        df = edges_df[edges_df.label == relation]
        from_id = df['from'].values.tolist()
        to_id = df['to'].values.tolist()
        return from_id, to_id

    def id2names(ids):  # 节点id得到节点（人物）的名字
        tables = []
        for ID in ids:
            tables.append(person_df[person_df['id'] == ID])
        names = pd.concat(tables)['label'].values.tolist()
        return names

    def get_relation(relation, graph):
        from_id, to_id = relation2id(relation)
        for from_label, to_label in zip(id2names(from_id), id2names(to_id)):
            # print(from_label, '--> {} -->'.format(relation), to_label)
            a = graph.nodes.match('Character', name=from_label).first()
            b = graph.nodes.match('Character', name=to_label).first()
            r = Relationship(a, relation, b)
            graph.create(r)

    # 根据关系查角色
    def search_char_by_relation(relation):
        data = test_graph.run('MATCH p=()-[r:`' + relation + '`]->() RETURN p LIMIT 25')
        for each in data:
            print(each)

    for each in char_list:  # 创建节点
        a = Node('Character', name=each)
        test_graph.create(a)

    for each in relation_list:
        get_relation(each, test_graph)

    search_char_by_relation("仆人")


relation_graph()


def author():
    """
    根据每一章节的关键词判断作者是否属于同一个人
    方法：K-means聚类
    参考文章ref: https://juejin.im/post/5b09945cf265da0dc562f316

    提取关键词后（每章1000个），并用sklearn中方法的向量化
    聚类后使用PCA降维（三维），用matplotlib可视化聚类效果
    :return:
    """
    character_list = []
    with open('character.txt', encoding='utf-8')as r:
        lines = r.readlines()
    for line in lines:
        name = line.split(' ')[0]
        character_list.append(name)

    word_feature = []
    for i in range(120):
        with open('chapters/' + str(i + 1) + '.txt', encoding='utf-8')as r:
            text = r.read()
        word_per_chapter = []
        topk = jieba.analyse.extract_tags(text, topK=1500)  # 不能统计整部小说的词频，因为前80比后40多了40回
        for j in range(len(topk)):
            if topk[j] not in character_list:  # 去掉角色名，因为剧情前后变化会对内容有影响
                word_per_chapter.append(topk[j])
        word_feature.append(word_per_chapter[:1000])
    word_feature = [" ".join(i) for i in word_feature]
    vertorizer = CountVectorizer(max_features=5000)
    train_data_features = vertorizer.fit_transform(word_feature)
    train_data_features = train_data_features.toarray()
    kmeans = KMeans(n_clusters=2, random_state=0).fit(train_data_features[0:120])
    print('前80回:', kmeans.labels_[:80])
    print('后40回:', kmeans.labels_[80:])

    # pca降维，因为画图不能超过三维
    pca = PCA(n_components=3)
    new_train_data = pca.fit_transform(train_data_features)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(new_train_data[:, 0], new_train_data[:, 1], new_train_data[:, 2], c=kmeans.labels_, marker='o')
    for i in range(len(new_train_data)):
        ax.text(new_train_data[i, 0], new_train_data[i, 1], new_train_data[i, 2], str(i + 1))
    plt.savefig('cluster.jpg', dpi=300)
    plt.show()
