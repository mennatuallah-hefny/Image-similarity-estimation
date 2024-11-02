import tensorflow as tf
from keras import metrics
from model import create_embedding_model

def evaluate_similarity(anchor, positive, negative):
    embedding = create_embedding_model()
    cosine_similarity = metrics.CosineSimilarity()
    anchor_embedding = embedding(anchor)
    positive_embedding = embedding(positive)
    negative_embedding = embedding(negative)

    positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
    negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
    
    print("Positive similarity:", positive_similarity.numpy())
    print("Negative similarity:", negative_similarity.numpy())
