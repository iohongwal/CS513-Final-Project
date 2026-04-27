from deepface import DeepFace

def predict_emotion(img):
    result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
    return result[0]