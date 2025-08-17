import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def get_agent_plan(user_prompt: str):
    """
    Menerima prompt dari pengguna dan meminta LLM untuk merencanakan
    tool mana yang harus digunakan.
    """
    available_tools = """
    1. describe: Berguna untuk mendapatkan ringkasan statistik dari dataset.
    2. correlation-heatmap: Berguna untuk membuat visualisasi korelasi antar kolom numerik.
    3. histogram: Berguna untuk membuat visualisasi distribusi dari satu kolom numerik tertentu.
    """

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """Anda adalah seorang AI data analyst expert bernama Data Whisperer.
        Tugas Anda adalah memahami permintaan pengguna dan memutuskan alat mana yang paling tepat untuk digunakan dari daftar berikut:
        {tools}

        Anda HARUS merespons HANYA dengan format JSON yang valid, tanpa teks atau penjelasan tambahan.
        Format JSON harus seperti ini:
        {{
            "tool_name": "nama_alat_yang_dipilih",
            "reasoning": "alasan singkat mengapa Anda memilih alat tersebut",
            "column_name": "nama_kolom_jika_dibutuhkan_oleh_alat_histogram"
        }}
        
        Jika pengguna meminta histogram, Anda harus mengekstrak nama kolom dari prompt pengguna dan memasukkannya ke field "column_name". Jika alat lain yang dipilih, isi "column_name" dengan null.
        """),
        ("human", "{user_input}")
    ])

    chain = prompt_template | llm | StrOutputParser()

    try:
        response_str = chain.invoke({
            "tools": available_tools,
            "user_input": user_prompt
        })
        
        # --- LANGKAH DEBUGGING: Tampilkan respons mentah dari AI ---
        print("--- Raw Response from LLM ---")
        print(repr(response_str)) # Menggunakan repr() untuk melihat karakter tersembunyi
        print("-----------------------------")

        # --- SOLUSI: Bersihkan respons sebelum parsing ---
        # LLM seringkali membungkus JSON dengan ```json ... ``` atau ``` ... ```
        if "```json" in response_str:
            response_str = response_str.split("```json")[1].split("```")[0].strip()
        elif "```" in response_str:
             response_str = response_str.split("```")[1].strip()

        # Mengubah string JSON yang sudah bersih menjadi dictionary
        return json.loads(response_str)
        
    except Exception as e:
        print(f"Error saat memproses rencana agent: {e}")
        return {"error": "Gagal menghasilkan rencana dari LLM.", "detail": str(e)}