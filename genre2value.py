import pandas as pd
import os


table = {
'節句・年中行事'             :  '1'   ,
'神輿・山車など'             :  '2'   ,
'伝統芸能・舞踊'             :  '3'   ,
'行列・パレード'             :  '4'   ,
'食の祭り'                     :  '5'   ,
'市・縁日'                   :  '6'   ,
'花見・自然'                 :  '7'   ,
'火と灯の祭り'               :  '8'   ,
'花火大会'                   :  '9'   ,
'雪・冬祭り'                 :  '10'  ,
'イルミネーション'           :  '11'  ,
'音楽祭・映画祭'                 :  '12'  ,
'スポーツ'                   :  '13'  ,
'美術展・博物展'             :  '14'  ,
'博物館・美術館展'           :  '14'  ,
'フェスティバル'     :  '15'  ,
'動物の祭り'               :  '16'  ,
'体験イベント'               :  '17'  ,
'学園祭'                     :  '18'  ,
'講演会・トークショー'       :  '19'  ,
'演劇・舞台'                 :  '20'  ,
'テーマパーク'               :  '21'  ,
'動物園・水族館'             :  '22'  ,
'記念日'                     :  '23'  ,
'即売会・フェア'             :  '24'  ,
'その他'                     :  '25'  ,
'産業・商工祭'               :  '26'  ,
'習俗・風俗'                 :  '27'  ,
'農耕儀礼'                   : '28' ,
'0'                          : '0'
}

df = pd.read_csv('./rurubu.csv',delimiter=',')
events = df[df['kind'] == 'イベント']
genres = events['genre'].str.split('／',expand=True)
events['genre'] = genres[0]
events['genre2'] = genres[1]
events['genre2_num'] = events['genre2'].replace(table)
events['genre_num'] = events['genre'].replace(table)
events.to_csv('./genre_test_changed.txt')

df = pd.DataFrame()
df['text'] = events['event_name']
df['label'] = events['genre_num']

EXTRACTDIR = "."
df[:len(df) // 5].to_csv( os.path.join(EXTRACTDIR, "test.tsv"), sep='\t', index=False)
df[len(df) // 5:len(df)*2 // 5].to_csv( os.path.join(EXTRACTDIR, "dev.tsv"), sep='\t', index=False)
df[len(df)*2 // 5:].to_csv( os.path.join(EXTRACTDIR, "train.tsv"), sep='\t', index=False)

