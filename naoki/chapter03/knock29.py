import urllib
#'国旗画像'キーに対応するURLエンコード
url = 'https://www.mediawiki.org/w/api.php?action=query&titles=File:' + urllib.parse.quote(inf_dic4['国旗画像']) + '&format=json&prop=imageinfo&iiprop=url'
connection = urllib.request.urlopen(urllib.request.Request(url))
response = json.loads(connection.read().decode())
#response:辞書 ['query']:APIレスポンスのqueryを取得し、pageを取得させる。pagesの辞書内でキーが-1の要素を取得する。これは要求されたファイルに関する情報を含む特別なケースを指している。imageinfoはファイルに関するメタデータを含むリストである。
print(response['query']['pages']['-1']['imageinfo'][0]['url'])