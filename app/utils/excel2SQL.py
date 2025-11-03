import pandas as pd

"""
excel文件转换成各种类型文件 SQL、TXT
"""


class DealExcelAuthKey:
    def __init__(self, excel_path, output_path):
        self.excel_path = excel_path
        self.output_path = output_path

    """
    授权.xlsx 清洗成 法院授权配置.txt
    """

    def deal_excel(self):
        # 读取 Excel
        df = pd.read_excel(self.excel_path)

        code_list = []

        with open(self.output_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                code = row["法院代码"]
                name = row["法院名称"]
                app_id = row["ag-app-id"]
                key = row["ag-key"]
                secret = row["ag-secret"]

                code_list.append(f"{code}")

                f.write(f"# {code} {name}\n")
                f.write(f"onenetwork.configs.{code}.agAppId={app_id}\n")
                f.write(f"onenetwork.configs.{code}.agKey={key}\n")
                f.write(f"onenetwork.configs.{code}.agSecret={secret}\n")
                f.write(f"onenetwork.configs.{code}.baseUrl=http://142.2.239.4:18080/sdavplatform/yzw/common\n\n")

        print(f"授权配置文件：{self.output_path}")
        print(f"已授权法院代码：{code_list}")


class DealExcelInsertSql:
    def __init__(self, excel_path, output_path):
        self.excel_path = excel_path
        self.output_path = output_path

    """
    法庭名称编号.xlsx 清洗成 insert sql
    """

    def deal_excel(self):
        # 读取 Excel
        df = pd.read_excel(self.excel_path)

        # 统计法庭名称重复数量
        name_count_map = {}

        with open(self.output_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                ftbh = str(row["法庭编号"]).strip().replace(" ", "")  # 去掉所有空格
                ftmc = str(row["云庭互联网法庭名称"]).strip().replace(" ", "")  # 去掉所有空格
                court_name = str(row["法院名称"]).strip().replace(" ", "")  # 去掉所有空格

                # 修正：检查空值、NaN和"nan"字符串
                if pd.isna(row["法庭编号"]) or ftbh.lower() == "nan" or not ftbh:
                    continue

                # 创建组合key - 去掉所有空格
                composite_key = f"{court_name} => {ftmc}"

                # 统计重复数量
                if composite_key in name_count_map:
                    name_count_map[composite_key] += 1
                else:
                    name_count_map[composite_key] = 1

                f.write(
                    f"insert into t_court_place (ftbh, ftmc, court_name) values ('{ftbh}', '{ftmc}', '{court_name}');\n"
                )

            # 找出重复的法庭名称
        duplicate_names = {name: count for name, count in name_count_map.items() if count > 1}

        for name, count in duplicate_names.items():
            print(f"法庭名称: '{name}', 重复次数: {count}")

        print(f"insert sql文件已生成：{self.output_path}")


class DealExcelUpdateSql:
    def __init__(self, excel_path, output_path):
        self.excel_path = excel_path
        self.output_path = output_path

    """
    法庭名称编号.xlsx 清洗成 update sql
    """

    def deal_excel(self):
        # 读取 Excel
        df = pd.read_excel(self.excel_path)
        with open(self.output_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                ftbh = str(row["法庭编号"]).strip().replace(" ", "")  # 去掉所有空格
                ftmc = str(row["云庭互联网法庭名称"]).strip().replace(" ", "")  # 去掉所有空格
                court_name = row["法院名称"]

                # 修正：检查空值、NaN和"nan"字符串
                if pd.isna(row["法庭编号"]) or ftbh.lower() == "nan" or not ftbh:
                    continue

                f.write(
                    f"update t_trial_place set ftbh = '{ftbh}' where trial_place_name = '{ftmc}';\n"
                )

        print(f"update sql文件已生成：{self.output_path}")


if __name__ == '__main__':
    deal_excel_auth_key = DealExcelAuthKey("../documents/授权.xlsx", "../documents/法院授权配置.txt")
    deal_excel_auth_key.deal_excel()

    deal_excel_insert_sql = DealExcelInsertSql("../documents/云庭互联网法庭名称_0909.xlsx",
                                               "../documents/insert_云庭法庭编号名称_0909.sql")
    deal_excel_insert_sql.deal_excel()

    deal_excel_update_sql = DealExcelUpdateSql("../documents/云庭互联网法庭名称_0909.xlsx",
                                               "../documents/update_云庭法庭编号名称_0909.sql")
    deal_excel_update_sql.deal_excel()
