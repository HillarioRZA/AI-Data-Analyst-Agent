import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns

def get_csv_description(file_contents: bytes):
    try:
        buffer = io.StringIO(file_contents.decode('utf-8'))
        df = pd.read_csv(buffer)
        description = df.describe().to_dict()
        return description
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def generate_correlation_heatmap(file_contents: bytes):
    try:
        buffer = io.StringIO(file_contents.decode('utf-8'))
        df = pd.read_csv(buffer)
        df_numeric = df.select_dtypes(include=['number'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)
        return img_buffer.getvalue()
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return None

def generate_histogram(file_contents: bytes, column_name: str):
    """
    Menerima konten file CSV dan nama kolom, membuat gambar histogram,
    dan mengembalikannya dalam bentuk bytes.
    """
    try:
        buffer = io.StringIO(file_contents.decode('utf-8'))
        df = pd.read_csv(buffer)
        if column_name not in df.columns:
            return "column_not_found"
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            return "column_not_numeric"

        plt.figure(figsize=(10, 6))
        sns.histplot(df[column_name], kde=True)
        plt.title(f'Histogram of {column_name}', fontsize=16)
        plt.xlabel(column_name, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)

        return img_buffer.getvalue()

    except Exception as e:
        print(f"Error generating histogram: {e}")
        return None