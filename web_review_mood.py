from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import threading

app = Flask(__name__)

# Глобальные переменные для отслеживания состояния модели
classifier = None
model_loading = True
model_error = None


def load_model():
    """Загрузка модели в отдельном потоке"""
    global classifier, model_loading, model_error

    try:
        classifier = pipeline(
            "text-classification",
            model="./review_mood_model",
            tokenizer="./review_mood_tokenizer",
            top_k=None
        )
        model_loading = False
    except Exception as e:
        model_error = str(e)
        model_loading = False
        print(f"Ошибка загрузки модели: {e}")


# Запуск загрузки модели в фоновом режиме при старте приложения
threading.Thread(target=load_model, daemon=True).start()


@app.route('/')
def home():
    """Главная страница"""
    return render_template('index.html',
                           loading=model_loading, error=model_error)


@app.route('/status')
def status():
    """Эндпоинт для проверки статуса загрузки модели"""
    return jsonify({
        "loading": model_loading,
        "error": model_error
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """Анализ тональности текста"""
    if model_loading:
        return jsonify({
            "error": "Модель еще загружается. Пожалуйста, подождите."
            }), 503

    if model_error:
        return jsonify({
            "error": f"Ошибка загрузки модели: {model_error}"
            }), 500

    text = request.form.get('text', '').strip()

    if not text:
        return jsonify({"error": "Пожалуйста, введите текст для анализа"}), 400

    try:
        # Анализ тональности
        results = classifier(text)

        # Карта для преобразования LABEL_X в читаемые названия
        label_map = {
            "LABEL_0": "negative",
            "LABEL_1": "neutral",
            "LABEL_2": "positive"
        }

        # Форматирование результатов
        formatted_results = []
        for result in results[0]:
            formatted_results.append({
                "label": label_map.get(result["label"], result["label"]),
                "score": result["score"],
                "display_label": {
                    "negative": "Отрицательная",
                    "neutral": "Нейтральная",
                    "positive": "Положительная"
                }.get(label_map.get(
                    result["label"], result["label"]), result["label"])
            })

        # Определение доминирующего класса
        dominant = max(formatted_results, key=lambda x: x["score"])

        return jsonify({
            "success": True,
            "text": text,
            "dominant_sentiment": dominant["label"],
            "dominant_display": dominant["display_label"],
            "confidence": dominant["score"],
            "all_scores": formatted_results
        })

    except Exception as e:
        return jsonify({"error": f"Ошибка анализа: {str(e)}"}), 500


if __name__ == '__main__':
    # Запуск приложения
    print("Запуск сервера Flask")
    app.run(debug=False, host='0.0.0.0', port=5000)
