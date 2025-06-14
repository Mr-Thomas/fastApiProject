import camelot  # pip install camelot-py[cv]
import pandas as pd  # pip install pandas
from bs4 import BeautifulSoup  # pip install beautifulsoup4
import json


class HTML2Excel:
    def __init__(self, html_path, output_excel_path):
        """
        初始化html数据清理工具
        :param html_path: 输入html文件路径
        :param output_excel_path: 输出Excel文件路径
        """
        self.html_path = html_path
        self.output_excel_path = output_excel_path

    def extract_tables_with_soup(self):
        # 读取HTML文件
        with open(self.html_path, 'r', encoding='utf-8') as f:
            html = f.read()

        soup = BeautifulSoup(html, 'html.parser')

        # 查找表格
        table = soup.find('table')

        # 提取表头
        headers = [th.get_text(strip=True) for th in table.find('thead').find_all('th')]

        # 提取表格数据
        data = []
        for row in table.find('tbody').find_all('tr'):
            row_data = []
            for td in row.find_all('td'):
                # 处理单元格中的span标签并移除换行
                cell_text = ''.join([span.get_text(strip=True) for span in td.find_all('span')]) or td.get_text(
                    strip=True)
                # 移除所有换行符和多余空格
                cell_text = ''.join(cell_text.split())
                row_data.append(cell_text)
            data.append(row_data)

        # 创建DataFrame
        df = pd.DataFrame(data, columns=headers)

        # 处理json列表类型的数据
        json_columns = [col for col in df.columns if 'json列表' in df[col].iloc[0]]
        for col in json_columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if x.startswith('[') and x.endswith(']') else x)

        # 保存到Excel pip install openpyxl
        df.to_excel(self.output_excel_path, index=False, engine='openpyxl')


class PDFDataClean:
    def __init__(self, pdf_path, output_excel_path):
        """
        初始化PDF数据清理工具
        :param pdf_path: 输入PDF文件路径
        :param output_excel_path: 输出Excel文件路径
        """
        self.pdf_path = pdf_path
        self.output_excel_path = output_excel_path

    @staticmethod
    def clean_text(text):
        """
        清理单个单元格的文本：去除换行符、多余空格、特殊字符等
        :param text: 要清理的文本
        :return: 清理后的文本
        """
        if isinstance(text, str):  # 确保是字符串类型
            # 去除换行符 \n 和 \r
            text = text.replace('\n', '').replace('\r', '')
            # 去除多余空格（多个空格合并为1个）
            text = ' '.join(text.split())
            # 可选：去除其他特殊字符（如制表符 \t）
            text = text.replace('\t', '')
            return text.strip()  # 去除首尾空格
        return text  # 如果不是字符串（如数字），直接返回

    def clean_dataframe(self, df):
        """
        清理整个DataFrame的文本数据
        :param df: 要清理的DataFrame
        :return: 清理后的DataFrame
        """
        return df.map(self.clean_text)  # 对每个单元格应用clean_text函数

    def extract_tables_with_camelot(self):
        """
        使用camelot提取PDF中的表格，过滤第一行并清理数据，最后保存为Excel文件
        """
        try:
            # 提取所有表格
            tables = camelot.read_pdf(self.pdf_path, pages='all')

            if not tables:
                print("警告: 未从PDF中提取到任何表格!")
                return

            # 创建一个空的DataFrame列表，用于存储所有表格数据
            all_tables_df = []

            # 遍历所有表格
            for table in tables:
                # 将表格转换为pandas DataFrame
                df = table.df

                if len(df) > 1:  # 确保至少有2行才过滤
                    # 过滤掉第一行（假设第一行是表头或标题）
                    filtered_df = df.iloc[1:]  # 或者 df.drop(0, axis=0)
                else:
                    filtered_df = df  # 否则保留原表

                # 清理表格数据（去除换行符、多余空格等）
                cleaned_df = self.clean_dataframe(filtered_df)
                # 将表格添加到列表中
                all_tables_df.append(cleaned_df)

            if not all_tables_df:
                print("警告: 没有可合并的表格数据!")
                return

            # 将所有DataFrame合并为一个大的DataFrame
            # ignore_index=True 会重新生成索引，避免合并后出现重复的索引值
            merged_df = pd.concat(all_tables_df, ignore_index=True)

            # 将合并后的DataFrame保存为Excel文件
            merged_df.to_excel(self.output_excel_path, index=False)
            print(f"成功: 表格已保存到 {self.output_excel_path}")

        except Exception as e:
            print(f"错误: 处理PDF时发生异常 - {str(e)}")


if __name__ == '__main__':
    # 示例使用
    # pdf_file_path = "../documents/数据项列表-案由-06-06.pdf"
    # output_excel_path = "../documents/output_tables.xlsx"
    # 创建PDFDataClean实例
    # pdf_cleaner = PDFDataClean(pdf_file_path, output_excel_path)
    # 调用方法提取并清理表格
    # pdf_cleaner.extract_tables_with_camelot()

    html2Excel = HTML2Excel("../documents/数据项列表-案由-06-12.html", "../documents/html_output_tables.xlsx")
    html2Excel.extract_tables_with_soup()
