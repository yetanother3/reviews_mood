from transformers import pipeline
import argparse
import json

def load_sentiment_analyzer(model_path="./sentiment_model", tokenizer_path="./sentiment_tokenizer"):
    """
    Загружает модель для анализа тональности с использованием pipeline
    """
    # Загрузка pipeline для классификации текста
    classifier = pipeline(
        "text-classification", 
        model=model_path,
        tokenizer=tokenizer_path,
        return_all_scores=True
    )
    return classifier

def analyze_sentiment(classifier, text):
    """
    Анализирует тональность переданного текста
    """
    # Получение предсказаний для всех классов
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
            "score": result["score"]
        })
    
    # Определение доминирующего класса
    dominant = max(formatted_results, key=lambda x: x["score"])
    
    return {
        "dominant_sentiment": dominant["label"],
        "confidence": dominant["score"],
        "all_scores": formatted_results
    }

def main():
    parser = argparse.ArgumentParser(description="Анализ тональности текста с помощью предобученной модели")
    parser.add_argument("--text", type=str, help="Текст для анализа")
    parser.add_argument("--file", type=str, help="Путь к файлу с текстами для анализа (по одному на строку)")
    parser.add_argument("--model_dir", type=str, default="./sentiment_model", help="Путь к директории с моделью")
    parser.add_argument("--tokenizer_dir", type=str, default="./sentiment_tokenizer", help="Путь к директории с токенизатором")
    parser.add_argument("--output", type=str, help="Путь к файлу для сохранения результатов в формате JSON")
    
    args = parser.parse_args()
    
    # Загрузка модели
    print("Загрузка модели для анализа тональности...")
    classifier = load_sentiment_analyzer(args.model_dir, args.tokenizer_dir)
    print("Модель успешно загружена!")
    
    results = []
    
    # Анализ одного текста
    if args.text:
        result = analyze_sentiment(classifier, args.text)
        results.append({"text": args.text, "result": result})
        print(f"\nТекст: {args.text}")
        print(f"Тональность: {result['dominant_sentiment']} (уверенность: {result['confidence']:.4f})")
        print("\nВсе оценки:")
        for score in result['all_scores']:
            print(f"  {score['label']}: {score['score']:.4f}")
    
    # Анализ текстов из файла
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"\nАнализ {len(texts)} текстов из файла {args.file}...")
            for i, text in enumerate(texts, 1):
                result = analyze_sentiment(classifier, text)
                results.append({"text": text, "result": result})
                
                print(f"\n[{i}/{len(texts)}] Текст: {text[:50]}{'...' if len(text) > 50 else ''}")
                print(f"Тональность: {result['dominant_sentiment']} (уверенность: {result['confidence']:.4f})")
        except FileNotFoundError:
            print(f"Ошибка: файл {args.file} не найден")
            return
    
    # Интерактивный режим
    else:
        print("\nИнтерактивный режим анализа тональности")
        print("Введите текст для анализа (или 'exit' для выхода):")
        
        while True:
            text = input("\n> ")
            if text.lower() in ['exit', 'quit', 'выход']:
                break
            
            result = analyze_sentiment(classifier, text)
            print(f"Тональность: {result['dominant_sentiment']} (уверенность: {result['confidence']:.4f})")
            print("Детальные оценки:")
            for score in result['all_scores']:
                print(f"  {score['label']}: {score['score']:.4f}")
            
            results.append({"text": text, "result": result})
    
    # Сохранение результатов в файл
    if args.output and results:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nРезультаты сохранены в файл {args.output}")

if __name__ == "__main__":
    main()