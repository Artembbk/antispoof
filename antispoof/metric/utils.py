import Levenshtein

# Don't forget to support cases when target_text == ''

def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 0
    # Преобразование текстов в списки символов
    predicted_chars = list(predicted_text)
    target_text = list(target_text)

    # Рассчитайте расстояние Левенштейна между последовательностями символов
    distance = Levenshtein.distance(predicted_text, target_text)

    # Рассчитайте CER
    cer_score = distance / len(target_text)

    return cer_score
    


def calc_wer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 0
    predicted_text = predicted_text.split()
    target_text = target_text.split()

    distance = Levenshtein.distance(predicted_text, target_text)
    wer_score = distance / len(target_text)

    

    return wer_score