import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import openai
from typing import List, Dict, Optional
import re
import json
from datetime import datetime
import argparse

class LawVectorizer:
    """法令ベクトル化専用クラス"""

    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.law_metadata = {}
        self.chunks = []

    def load_law_metadata(self, csv_file_path: str):
        """CSVファイルから法令メタデータを読み込み"""
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        df = df.dropna(subset=['法令ID', '法令名'])

        for _, row in df.iterrows():
            law_id = row['法令ID']
            self.law_metadata[law_id] = {
                'law_type': row['法令種別'],
                'law_number': row['法令番号'],
                'law_name': row['法令名'],
                'law_name_kana': row['法令名読み'],
                'publication_date': row['公布日'],
                'enforcement_date': row['施行日'],
                'url': row['本文URL'],
                'not_enforced': row['未施行'] == '○'
            }

        print(f"法令メタデータ {len(self.law_metadata)} 件を読み込みました")

    def analyze_zeihou_folder(self, zeihou_folder_path: str):
        """zeihouフォルダの構造を分析"""
        print(f"\nzeihouフォルダ構造分析: {zeihou_folder_path}")

        items = os.listdir(zeihou_folder_path)
        xml_files = [f for f in items if f.endswith('.xml')]
        folders = [d for d in items if os.path.isdir(os.path.join(zeihou_folder_path, d))]

        print(f"項目数: {len(items)}")
        print(f"XMLファイル数: {len(xml_files)}")
        print(f"サブフォルダ数: {len(folders)}")

        if folders:
            print(f"サブフォルダ例: {folders[:5]}")
            for sample_dir in folders[:3]:
                sample_path = os.path.join(zeihou_folder_path, sample_dir)
                try:
                    sample_files = os.listdir(sample_path)
                    xml_in_folder = [f for f in sample_files if f.endswith('.xml')]
                    print(f"  {sample_dir}/: {len(sample_files)}ファイル (XML: {len(xml_in_folder)})")
                except:
                    print(f"  {sample_dir}/: アクセスエラー")

    def find_xml_files(self, zeihou_folder_path: str, law_id: str) -> List[str]:
        """XMLファイルを検索"""
        possible_paths = [
            os.path.join(zeihou_folder_path, law_id, f"{law_id}.xml"),
            os.path.join(zeihou_folder_path, f"{law_id}.xml"),
        ]

        return [path for path in possible_paths if os.path.exists(path)]

    def parse_law_xml(self, xml_file_path: str) -> List[Dict]:
        """XMLファイルを解析して条文を抽出"""
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            # 法令名を取得
            law_name = ""
            law_name_elem = root.find('.//法令名')
            if law_name_elem is not None:
                law_name = law_name_elem.text or ""

            articles = []

            # 条文を抽出
            for article in root.findall('.//条'):
                article_data = {
                    'law_name': law_name,
                    'article_number': article.get('Num', ''),
                    'title': '',
                    'paragraphs': []
                }

                # 条文タイトル
                title_elem = article.find('条題')
                if title_elem is not None:
                    article_data['title'] = title_elem.text or ''

                # 各項を抽出
                for paragraph in article.findall('.//項'):
                    paragraph_text = self._extract_text_from_element(paragraph)
                    if paragraph_text:
                        article_data['paragraphs'].append(paragraph_text)

                if article_data['paragraphs']:
                    articles.append(article_data)

            return articles

        except Exception as e:
            print(f"XML解析エラー {xml_file_path}: {e}")
            return []

    def _extract_text_from_element(self, element) -> str:
        """XML要素からテキストを抽出"""
        text = ET.tostring(element, encoding='unicode', method='text')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process_laws(self, zeihou_folder_path: str, target_laws: Optional[List[str]] = None):
        """法令を処理してチャンクを生成"""
        if not os.path.exists(zeihou_folder_path):
            print(f"zeihouフォルダが見つかりません: {zeihou_folder_path}")
            return

        self.analyze_zeihou_folder(zeihou_folder_path)

        all_chunks = []
        processed_count = 0

        if target_laws is None:
            target_laws = [
                "所得税法", "法人税法", "消費税法", "相続税法", "贈与税法",
                "地方税法", "租税特別措置法", "国税通則法", "国税徴収法"
            ]

        print(f"\n処理対象法令: {', '.join(target_laws)}")

        for law_id, metadata in self.law_metadata.items():
            if not any(target_law in metadata['law_name'] for target_law in target_laws):
                continue

            xml_paths = self.find_xml_files(zeihou_folder_path, law_id)
            if not xml_paths:
                print(f"XMLファイルが見つかりません: {law_id} ({metadata['law_name']})")
                continue

            xml_file_path = xml_paths[0]
            print(f"処理中: {metadata['law_name']}")

            articles = self.parse_law_xml(xml_file_path)
            if not articles:
                print(f"  ⚠️ 条文が抽出できませんでした")
                continue

            # チャンクに変換
            for article in articles:
                chunks = self._create_chunks_from_article(article, metadata, law_id)
                all_chunks.extend(chunks)

            processed_count += 1
            law_chunks = [c for c in all_chunks if c['metadata']['law_name'] == metadata['law_name']]
            print(f"  ✅ 完了: {len(law_chunks)}チャンク生成")

        self.chunks = all_chunks
        print(f"\n総チャンク数: {len(self.chunks)}")

        # 処理された法令の一覧を表示
        processed_laws = set(chunk['metadata']['law_name'] for chunk in self.chunks)
        print(f"\n処理された法令一覧 ({len(processed_laws)}件):")
        for law_name in sorted(processed_laws):
            law_chunks = [c for c in self.chunks if c['metadata']['law_name'] == law_name]
            print(f"- {law_name}: {len(law_chunks)}チャンク")

    def _create_chunks_from_article(self, article: Dict, metadata: Dict, law_id: str) -> List[Dict]:
        """条文からチャンクを生成"""
        chunks = []

        base_info = {
            'law_id': law_id,
            'law_name': metadata['law_name'],
            'law_type': metadata['law_type'],
            'article_number': article['article_number'],
            'article_title': article['title'],
            'url': metadata['url']
        }

        # 各項をチャンクとして処理
        for i, paragraph in enumerate(article['paragraphs']):
            chunk_text = self._format_chunk_text(
                metadata['law_name'],
                article['article_number'],
                article['title'],
                i + 1,
                paragraph
            )

            chunk = {
                'id': f"{law_id}_{article['article_number']}_{i+1}",
                'text': chunk_text,
                'metadata': {
                    **base_info,
                    'paragraph_number': i + 1,
                    'paragraph_text': paragraph
                }
            }

            chunks.append(chunk)

        return chunks

    def _format_chunk_text(self, law_name: str, article_num: str, title: str, paragraph_num: int, text: str) -> str:
        """チャンクテキストをフォーマット"""
        formatted = f"【{law_name}】\n"
        formatted += f"第{article_num}条"

        if title:
            formatted += f"（{title}）"

        formatted += f"\n第{paragraph_num}項: {text}"

        return formatted

    def create_embeddings(self, batch_size: int = 50, model: str = "text-embedding-3-large"):
        """エンベディングを生成"""
        if not self.chunks:
            print("チャンクが存在しません。先にprocess_laws()を実行してください。")
            return None

        texts = [chunk['text'] for chunk in self.chunks]
        embeddings = []

        print(f"エンベディング生成開始... ({len(texts)} チャンク)")
        print(f"使用モデル: {model}")

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            try:
                response = self.client.embeddings.create(
                    model=model,
                    input=batch
                )

                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

                print(f"進捗: {min(i + batch_size, len(texts))}/{len(texts)}")

            except Exception as e:
                print(f"エンベディング生成エラー (batch {i//batch_size + 1}): {e}")
                continue

        embeddings_array = np.array(embeddings)
        print(f"エンベディング生成完了: {embeddings_array.shape}")

        return embeddings_array

    def save_vectorized_data(self, embeddings: np.ndarray, output_dir: str = "./vectorized_data"):
        """ベクトル化データを保存"""
        os.makedirs(output_dir, exist_ok=True)

        # チャンクデータ
        chunks_file = os.path.join(output_dir, "chunks.json")
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        # エンベディング
        embeddings_file = os.path.join(output_dir, "embeddings.npy")
        np.save(embeddings_file, embeddings)

        # メタデータ
        metadata_file = os.path.join(output_dir, "metadata.json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.law_metadata, f, ensure_ascii=False, indent=2)

        # インデックス情報
        index_info = {
            "created_at": datetime.now().isoformat(),
            "total_chunks": len(self.chunks),
            "embedding_model": "text-embedding-3-large",
            "embedding_dimension": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
            "processed_laws": list(set(chunk['metadata']['law_name'] for chunk in self.chunks))
        }

        index_file = os.path.join(output_dir, "index_info.json")
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index_info, f, ensure_ascii=False, indent=2)

        print(f"ベクトル化データを {output_dir} に保存しました")
        print(f"- チャンクデータ: {chunks_file}")
        print(f"- エンベディング: {embeddings_file}")
        print(f"- メタデータ: {metadata_file}")
        print(f"- インデックス情報: {index_file}")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="法令ベクトル化処理")
    parser.add_argument("--csv", default="13.csv", help="CSVファイルのパス")
    parser.add_argument("--zeihou", default="./zeihou", help="zeihouフォルダのパス")
    parser.add_argument("--output", default="./vectorized_data", help="出力ディレクトリ")
    parser.add_argument("--api-key", help="OpenAI API キー")
    parser.add_argument("--laws", nargs="*", help="処理対象法令（指定しない場合は全ての税法）")
    parser.add_argument("--batch-size", type=int, default=50, help="バッチサイズ")

    args = parser.parse_args()

    # API キー取得
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("OpenAI API キーを入力してください: ")

    if not api_key:
        print("API キーが必要です")
        return

    # ファイル存在確認
    if not os.path.exists(args.csv):
        print(f"CSVファイルが見つかりません: {args.csv}")
        return

    if not os.path.exists(args.zeihou):
        print(f"zeihouフォルダが見つかりません: {args.zeihou}")
        return

    # ベクトル化処理実行
    print("=== 法令ベクトル化処理開始 ===")

    vectorizer = LawVectorizer(api_key)

    print("1. メタデータ読み込み...")
    vectorizer.load_law_metadata(args.csv)

    print("2. XML解析・チャンク生成...")
    target_laws = args.laws if args.laws else None
    vectorizer.process_laws(args.zeihou, target_laws)

    if not vectorizer.chunks:
        print("チャンクが生成されませんでした。処理を終了します。")
        return

    print("3. エンベディング生成...")
    embeddings = vectorizer.create_embeddings(batch_size=args.batch_size)

    if embeddings is None:
        print("エンベディング生成に失敗しました。")
        return

    print("4. データ保存...")
    vectorizer.save_vectorized_data(embeddings, args.output)

    print("=== ベクトル化処理完了 ===")


if __name__ == "__main__":
    main()