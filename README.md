# 税法検索システム

このリポジトリは、日本の税法に関する XML データをベクトル化し、検索・質問応答が行える API および Web アプリを提供します。

## ディレクトリ構成

- `zeihou/` - e-Gov から取得した大量の税法 XML データが格納されています。
- `13.csv` - 法令メタデータの CSV ファイル。
- `vectorizer.py` - XML からテキストを抽出し OpenAI Embedding を生成するスクリプト。
- `api.py` - 検索や質問応答を提供する FastAPI サーバー。
- `app.py` - Streamlit で実装されたフロントエンド。

## 事前準備

1. Python 3.10 以上をインストールしてください。
2. OpenAI API キーを取得し、環境変数 `OPENAI_API_KEY` に設定します。
3. `pip` で必要なライブラリをインストールします。

```bash
pip install openai fastapi uvicorn[standard] pandas numpy streamlit plotly requests
```

## ベクトル化データの作成

大量の XML ファイルを処理するため、`zeihou/` フォルダには数百 MB 規模のデータが入ります。ディスク容量に注意してください。

以下のコマンド例では、XML から条文を抽出しベクトル化データを `vectorized_data/` に保存します。

```bash
python vectorizer.py --csv 13.csv --zeihou ./zeihou --output ./vectorized_data --api-key $OPENAI_API_KEY
```

`--laws` オプションを指定すると処理対象の法令を絞り込むことができます。

## API サーバーの起動

ベクトル化後のデータを利用して FastAPI サーバーを起動します。

```bash
OPENAI_API_KEY=$OPENAI_API_KEY VECTORIZED_DATA_DIR=./vectorized_data uvicorn api:app --reload
```

`/search` や `/question` エンドポイントから検索・質問応答が利用できます。

## Streamlit アプリの起動

フロントエンドの Web アプリは `app.py` で提供されています。別ターミナルから次のように実行してください。

```bash
streamlit run app.py
```

デフォルトでは `http://localhost:8000` の API を参照します。必要に応じて `TaxLawClient` の `base_url` を変更してください。

## 参考

- e-Gov 法令データ提供システム <https://www.e-gov.go.jp/>
- OpenAI API <https://platform.openai.com/>

