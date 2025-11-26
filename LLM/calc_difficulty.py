import lib.annotation.Q_Extract as qe
import lib.annotation.SampleSelf_Consistency as ssc
import sys


def main():
    print ("start main!")
     
    t_extract = qe.Q_Extract(2)
    df = t_extract.db_extract()
    q_output = t_extract.tb_extract(df)

    print('SampleSelf_Consistency start')
    sample_sc = ssc.SampleSelf_Consistency('2024-12-04', 1, q_output[:1]) 
    print('SampleSelf_Consistency end')

    print('write_promt start')
    sample_sc.write_promt()
    print('write_promt end')

    print('calc_acc_for_l start')
    sample_sc.calc_acc_for_l()
    print('calc_acc_for_l end')

    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            print(f"Argument {i}: {arg}")
    main()

