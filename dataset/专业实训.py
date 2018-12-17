import os
import pandas as pd


# ȥ�ظ�ֵ
def Deduplication(filein, fileout):
    df = pd.read_csv(filein)
    print(df.apply(pd.value_counts))
    df.apply(pd.value_counts).to_csv(fileout)


# ������ļ��е�API�ϲ���һ���ļ�
def merge():
    SaveFile_Path = r'd:\abcdefg'

    SaveFile_Name = r'0.csv'

    file_list = os.listdir(SaveFile_Path)

    # df = pd.read_csv(SaveFile_Path +'\\'+ file_list[0],encoding="gbk")
    #
    # df.to_csv(SaveFile_Path+'\\'+ SaveFile_Name,index=False)

    for i in range(1, len(file_list)):
        try:
            df = pd.read_csv(SaveFile_Path + '\\' + file_list[i], encoding="gbk")
            df.to_csv(SaveFile_Path + '\\' + SaveFile_Name, index=False, header=False, mode='a+')
        except:
            print(file_list[i])


# ��ÿ��������ʹ�õ�API���������һ���ļ�
def srip():
    filepath = r"..\Good"
    data = []
    # print(os.listdir(filepath))

    files = os.listdir(filepath)

    for filename in files:
        b = filepath + '\\' + filename
        if (os.path.isdir(b)):
            a = os.listdir(b)

            for exe in a:
                isfile = os.path.isfile(filepath + '\\' + filename + '\\' + exe)
                if (isfile):

                    dataset = pd.read_csv(filepath + '\\' + filename + '\\' + exe, encoding='gbk')

                    # dataset=pd.read_csv(filepath+'\\'+exe,encoding='gbk')
                    if dataset.notnull:
                        col = dataset.columns

                        for content in dataset[col[3]]:
                            data.append(content)
                        df = pd.DataFrame(data=data, columns=['api'])  # ��װ��pandas������֡��

                        df = df.drop_duplicates(['api'])
                        # os.remove(b+'\\'+exe)
                        df.to_csv("d:\\abcdefg" + "\\" + exe, index=False, header=False, na_rep='NA', mode='a+')
                        data = []

                        # print(df_fre)


# Ƶ�ʼ���
def frequent_compute():
    filepath = r"D:\abcdefg"
    data = []  # Ƶ��ͳ��
    data1 = []  # ���������
    # print(os.listdir(filepath))

    files = os.listdir(filepath)

    raw = pd.read_csv(r"C:\Users\60911\Desktop\1\sort0.csv", encoding='gbk')  # �����õ��ĺ����б����ظ���

    print(raw.columns)
    for filename in files:

        isfile = os.path.isfile(filepath + '\\' + filename)
        if (isfile):

            dataset = pd.read_csv(filepath + '\\' + filename, encoding='gbk')

            if dataset.notnull:
                col = dataset.columns
                data.append(filename)
                for r_content in raw.columns:  # ��ѭ������01���
                    count = 0  # ѭ�����Ʊ���
                    for content in dataset[col[0]]:  # ��ÿ���ļ���ͳ�ƣ��е���Ӧλ����1��û������0
                        count = count + 1
                        if r_content == content:
                            data.append(1)
                            break
                        elif count == len(dataset[col[0]]) - 1:
                            data.append(0)
                data1.append(data)
                df = pd.DataFrame(data=data1)
                print(data1)
                df.to_csv(r"C:\Users\60911\Desktop\1\result2.csv", index=False, header=False, na_rep='NA', mode='a+')
                # df = df.drop_duplicates(['api'])

                # df.to_csv(r"C:\Users\60911\Desktop\1\sorted.csv", index=False, header=False, na_rep='NA', mode='a+')
                data = []


# �ڴ˴������Լ����㷨�߼�
def main():
    print()


if __name__ == "__main__":
    main()
