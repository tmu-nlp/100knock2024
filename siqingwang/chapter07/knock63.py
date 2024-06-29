# knock63

vector_spain = model['Spain']
vector_madrid = model['Madrid']
vector_athens = model['Athens']

result_vector = vector_spain - vector_madrid + vector_athens

model.similar_by_vector(result_vector, topn=10)