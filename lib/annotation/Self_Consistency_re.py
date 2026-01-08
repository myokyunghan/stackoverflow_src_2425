from lib.annotation.import_files import *
from lib.annotation.VLLM import VLLM
import logging
# https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/Prompt_Engineering_with_Llama_3.ipynb
class Self_Consistency_re:
    def __init__(self, llm_model, few_shot_n, test_n, q_src_yn, ver, p_ver, sf_num, temperature, excel_ver, i):  
        self.ollama         = 'llama-3.1-70b-instruct-lorablated.Q4_K_M:latest'

        self.chatgpt        = OpenAI(api_key= conf.OEPN_AI_KEY)

        self.df             = pd.DataFrame()
        self.eval_prompt    = []
        self.result         = []
        self.message_list   = []
        self.eval_q_list    = []

        # param
        self.test_n         = test_n
        self.sys_prompt     = prompt[p_ver] 
        self.p_ver          = p_ver
        self.version        = str(ver)
        self.sf_num         = sf_num
        self.temperature    = temperature
        self.excel_ver      = excel_ver
        self.loop_i         = i
        self.tk             = AutoTokenizer.from_pretrained(conf.VLLM_CONF[llm_model]['model'], use_fast=True)
        # self.tk_max_length  = conf.VLLM_CONF[llm_model]['max_length']
        # print(f">>>>>>>>>>>>>>>>>>>>>conf.VLLM_CONF[llm_model]['model'] : {conf.VLLM_CONF[llm_model]['model']}")
        # log setting
        self.logger         = get_userlogger()
        self.logger.setLevel(logging.INFO)

        # init the process
        self.get_annotation_data(q_src_yn)
        e_f_dict = self.random_selection(few_shot_n, test_n)
        self.write_prompt(e_f_dict, few_shot_n)
        self.calc_acc(llm_model, few_shot_n, q_src_yn)


    def chk_max_length(self, message):
        prompt = self.tk.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_tokens = len(self.tk.encode(prompt))

        MAX_CONTEXT = self.tk.model_max_length
        MAX_GENERATION = 256
        SAFETY_MARGIN = 128

        tot_promt_tk = prompt_tokens + MAX_GENERATION + SAFETY_MARGIN
        return (tot_promt_tk > MAX_CONTEXT)

            
    def get_annotation_data(self, q_src_yn):
        file_path = f'{conf.DATA_PATH}/data/q_output'
        
        if q_src_yn == "Y":
            file_path = f'{file_path}_code_y'
        
        file_path = f'{file_path}{excel[self.excel_ver]}'
        self.df = pd.read_csv(f'{file_path}.csv')


    def set_fewshot_example(self, eval_q_id, few_shot_n):
        # hard coding for test :  
        # diff_idx = {  '<Difficulty Level>0</Difficulty Level>' : [72422859],
        #                 '<Difficulty Level>1</Difficulty Level>' : [72118859],
        #                 '<Difficulty Level>2</Difficulty Level>' : [76779313]
        #             }
        
        diff_idx = {x : np.setdiff1d(list(self.df[self.df['answer']==x].id), [eval_q_id]) for x in list(conf.DIFF_DICT.values())}

        fewshot_q_list = []
        for key, value in diff_idx.items():
            diff_population = value
            fewshot_q_list.append(np.random.choice(diff_population, size=few_shot_n, replace=True))
            self.logger.info(f'>>>>>>>>>>>>>>>! Self_Consistency re set_fewshot_example {fewshot_q_list}')
        return np.concatenate(fewshot_q_list)

    def random_selection(self, few_shot_n, test_n):
        # to evaluate self-consistency, pick eval target first
        # hard coding for test :  eval_q_id_list = [71389500]

        eval_q_id_list      = np.random.choice(list(self.df.id), size=test_n, replace=False)
        self.logger.info(f'>>>>>>>>>>>>>>>! Self_Consistency re random_selection {eval_q_id_list}')
 
        diff_s_idx = {}
        for eval_q_id in eval_q_id_list:
            diff_s_idx[eval_q_id] = dict()
            for sf_idx in range(self.sf_num):
                fewshot_q_list = self.set_fewshot_example(eval_q_id, few_shot_n)
                diff_s_idx[eval_q_id][sf_idx] = fewshot_q_list

        return diff_s_idx
            

    def write_prompt(self, e_f_dict, few_shot_n) : 
        
        # write system prompt & examples
        for eval_id, fewshot_dict in e_f_dict.items() : 
    
            for sc_idx, fewshot_id_list in fewshot_dict.items() : 
                message = []
                message.append({"role": "system", "content": self.sys_prompt})
                self.eval_q_list.append(eval_id)

                for fewshot_id in fewshot_id_list : 
                    

                    q_string = self.df.loc[self.df['id'] == fewshot_id, 'question'].iloc[0]
                    a_string = self.df.loc[self.df['id'] == fewshot_id, 'answer'].iloc[0]
                    t_string = self.df.loc[self.df['id'] == eval_id,    'question'].iloc[0]

                    q_prompt = """\nHere is the examples of question\n"""
                    q_prompt = q_prompt + q_string
                    
                    message.append({"role": "user", "content": q_prompt})
                    message.append({"role": "assistant", "content": a_string})
                
                target_post="""\nHere is the target post. Answer the "Difficulty Level".\n"""
                target_post = target_post+"""\n<target_post>\n"""
                target_post = target_post+t_string+'\n'
                target_post = target_post+"""</target_post>\n"""
    
                message.append({"role": "user", "content": target_post})

                self.logger.info(f'>>>>>>>>>>>>>>>! Self_Consistency re : {message}')
                
                if self.chk_max_length(message) :
                    e_f_dict[eval_id][sc_idx] = self.set_fewshot_example(eval_id, few_shot_n)
                    self.write_prompt(e_f_dict, few_shot_n)
                else :
                    self.message_list.append(message)



    def calc_acc_for_v(self, llm_model, few_shot_n, q_src_yn):
        self.logger.info(f'>>>>>>>>>>>>>>>calc_acc_for_v start!')
        self.logger.info(f'>>>>>>>>>>>>>>>calc_acc_for_v, load VLLM!')
        for idx, message in tqdm(enumerate(self.message_list)):
            tmp = []
            response = self.vllm.llm.chat(message, sampling_params=self.vllm.params) 

            tmp.append(self.eval_q_list[idx])
            tmp.append(response[0].outputs[0].text)
            self.result.append(tmp)
        result_df = pd.DataFrame(self.result, columns = ['id', 'result'])
        result_df = pd.merge(self.df, result_df, on = 'id')
        self.logger.info(f'>>>>>>>>>>>>>>>calc_acc_for_v savefile! {conf.DATA_PATH}{conf.ANNO_RESULT}/sc_{llm_model}_result_{few_shot_n}_{self.test_n}_{q_src_yn}_{self.version}_{self.p_ver}_{self.sf_num}_{self.temperature}_{self.excel_ver}_{self.loop_i}.csv')
        result_df.to_csv(f'{conf.DATA_PATH}{conf.ANNO_RESULT}/sc_{llm_model}_result_{few_shot_n}_{self.test_n}_{q_src_yn}_{self.version}_{self.p_ver}_{self.sf_num}_{self.temperature}_{self.excel_ver}_{self.loop_i}.csv')
        self.logger.info(f'>>>>>>>>>>>>>>>calc_acc_for_v end!')



 
    def calc_acc_for_l(self, llm_model, few_shot_n, q_src_yn):           
        for idx, message in tqdm(enumerate(self.message_list)):
            tmp = []
            response = chat( model      = self.ollama,
                            messages    = message,
                            )
            tmp.append(self.eval_q_list[idx])
            tmp.append(message)
            tmp.append(response['message']['content'])
            self.result.append(tmp)
        result_df = pd.DataFrame(self.result, columns = ['id', 'message', 'result'])
        result_df = pd.merge(self.df, result_df, on = 'id')
        
        result_df.to_csv(f'{conf.DATA_PATH}{conf.ANNO_RESULT}/sc_{llm_model}_result_{few_shot_n}_{self.test_n}_{q_src_yn}_{self.version}_{self.p_ver}_{self.sf_num}_{self.temperature}_{self.excel_ver}_{self.loop_i}.csv')

    def calc_acc_for_c(self, llm_model, few_shot_n, q_src_yn):
        for idx, message in tqdm(enumerate(self.message_list)):
            tmp = []
            MODEL = "gpt-4o"
            response = self.chatgpt.chat.completions.create(
                model=MODEL,
                messages=message,
                temperature= self.temperature,
            )
            tmp.append(self.eval_q_list[idx])
            tmp.append(message)
            tmp.append([response.choices[0].message.content])
            self.result.append(tmp)
        result_df = pd.DataFrame(self.result, columns = ['id', 'message', 'result'])
        result_df = pd.merge(self.df, result_df, on = 'id')
        result_df.to_csv(f'{conf.DATA_PATH}{conf.ANNO_RESULT}/sc_{llm_model}_result_{few_shot_n}_{self.test_n}_{q_src_yn}_{self.version}_{self.p_ver}_{self.sf_num}_{self.temperature}_{self.excel_ver}_{self.loop_i}.csv')


    def calc_acc(self, llm_model, few_shot_n, q_src_yn) :
        if llm_model == 'l' : # ollama 
            # print(self.eval_prompt)
            self.calc_acc_for_l(llm_model, few_shot_n, q_src_yn)
            

        elif llm_model == 'c' : # chatgpt 
            # print(self.eval_prompt)
            self.calc_acc_for_c(llm_model, few_shot_n, q_src_yn)

        elif llm_model == 'vl' : # vLLM + llama
            print("VLLM")
            self.vllm = VLLM(llm_model)
            self.calc_acc_for_v(llm_model, few_shot_n, q_src_yn)

        
        elif llm_model == 'vq' : # vLLM + qwen
            print("VLLM")
            self.vllm = VLLM(llm_model)
            self.calc_acc_for_v(llm_model, few_shot_n, q_src_yn)


def test(llm_model, few_shot_n, test_n, q_src_yn, ver, p_ver, sc_num, temperature, excel_ver):
    print(f"Test {llm_model}_{few_shot_n}_{test_n}_{q_src_yn}_{p_ver}_{sc_num} 시작")
    for i in range(ver):
        print(f"Test {llm_model}_{few_shot_n}_{test_n}_{q_src_yn}_{p_ver}_{sc_num} 실행 중: {i}")
        Self_Consistency_re(   llm_model
                            , few_shot_n
                            , test_n
                            , q_src_yn
                            , ver
                            , p_ver
                            , sc_num
                            , temperature
                            , excel_ver
                            , i)
    
    print(f"Task {llm_model}_{few_shot_n}_{test_n}_{q_src_yn}_{p_ver}_{sc_num} 완료")

if __name__ == "__main__":


    test ('vl',              # llm_model
        3,                # few_shot_n
        100,                # test_n(# of question for test)
        'Y',              # q_src_yn 
        50,                # iteration num
        'sys_prompt10',   # prompt ver
        5,                # self-consistency number
        0.01,             # temperature
        'ver7'            # excel_verion
        )

    # test ('vl',              # llm_model
    #         1,                # few_shot_n
    #         1,                # test_n(# of question for test)
    #         'Y',              # q_src_yn 
    #         1,                # iteration num
    #         'sys_prompt10',   # prompt ver
    #         1,                # self-consistency number
    #         0.01,             # temperature
    #         'ver7'            # excel_verion
    #         )
