from pythonrouge.pythonrouge import Pythonrouge

# truyền vào type_score = 'ROUGE-1' or 'ROUGE-2' or 'ROUGE-L'
def score_rouge(summary_system_path, summary_reference_path, type_score):
    # system summary(predict) & reference.txt summary
    with open(summary_system_path, 'r', encoding='-utf8') as f:
        sum_text = f.read()

    with open(summary_reference_path, 'r', encoding='-utf8') as f:
        ref_text = f.read()

    summary = [[sum_text]]
    reference = [[[ref_text]]]

    # initialize setting of ROUGE to eval ROUGE-1, 2, SU4
    # if you evaluate ROUGE by sentence list as above, set summary_file_exist=False
    # if recall_only=True, you can get recall scores of ROUGE
    rouge = Pythonrouge(summary_file_exist=False,
                        summary=summary, reference=reference,
                        n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                        recall_only=True, stemming=True, stopwords=True,
                        word_level=True, length_limit=True, length=50,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)
    score = rouge.calc_score()
    return score[type_score]
