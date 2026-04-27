def generate_report(result):
    return {
        "predicted_emotion": result["dominant_emotion"],
        "confidence": max(result["emotion"].values()),
        "emotion_scores": result["emotion"]
    }