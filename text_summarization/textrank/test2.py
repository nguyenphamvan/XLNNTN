from pyrouge import Rouge155
from text_summarization.textrank import settings

r = Rouge155()

# set directories
r.system_dir = 'system/'
r.model_dir = 'model/'

# define the patterns
r.system_filename_pattern = '(\d+)_system.txt'
r.model_filename_pattern = '#ID#_model.txt'

# use default parameters to run the evaluation
output = r.convert_and_evaluate(split_sentences=True)

output_dict = r.output_to_dict(output)
print('rouge_1_recall = %s'%output_dict['rouge_1_recall'])
print('rouge_1_precision = %s'%output_dict['rouge_1_precision'])
print('rouge_1_f = %s'%output_dict['rouge_1_f_score'])
print()
print('rouge_2_recall = %s'%output_dict['rouge_2_recall'])
print('rouge_2_precision = %s'%output_dict['rouge_2_precision'])
print('rouge_2_f = %s'%output_dict['rouge_2_f_score'])
print()
print('rouge_l_recall = %s'%output_dict['rouge_l_recall'])
print('rouge_l_precision = %s'%output_dict['rouge_l_precision'])
print('rouge_l_f = %s'%output_dict['rouge_l_f_score'])
