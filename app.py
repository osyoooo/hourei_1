import streamlit as st
import requests
import json
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# ページ設定
st.set_page_config(
    page_title="税法検索システム",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #2E8B57;
    text-align: center;
    padding: 1rem 0;
    border-bottom: 3px solid #2E8B57;
    margin-bottom: 2rem;
}

.search-result {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #2E8B57;
    margin: 1rem 0;
}

.similarity-score {
    background-color: #e3f2fd;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-weight: bold;
}

.law-name {
    color: #1976d2;
    font-weight: bold;
    font-size: 1.1rem;
}

.article-info {
    color: #666;
    font-style: italic;
}

.answer-box {
    background-color: #f0f8ff;
    padding: 1.5rem;
    border-radius: 10px;
    border: 2px solid #4169e1;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

class TaxLawClient:
    """バックエンドAPIクライアント"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self) -> bool:
        """ヘルスチェック"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_laws(self) -> List[str]:
        """利用可能な法令一覧を取得"""
        try:
            response = requests.get(f"{self.base_url}/laws")
            if response.status_code == 200:
                return response.json()["laws"]
            return []
        except:
            return []
    
    def get_system_info(self) -> Dict:
        """システム情報を取得"""
        try:
            response = requests.get(f"{self.base_url}/system-info")
            if response.status_code == 200:
                return response.json()
            return {}
        except:
            return {}
    
    def search(self, query: str, top_k: int = 5, law_filter: Optional[List[str]] = None) -> Dict:
        """検索実行"""
        try:
            payload = {
                "query": query,
                "top_k": top_k,
                "law_filter": law_filter
            }
            response = requests.post(f"{self.base_url}/search", json=payload)
            if response.status_code == 200:
                return response.json()
            return {"error": f"検索エラー: {response.status_code}"}
        except Exception as e:
            return {"error": f"接続エラー: {str(e)}"}
    
    def ask_question(self, question: str, top_k: int = 3, law_filter: Optional[List[str]] = None) -> Dict:
        """質問応答実行"""
        try:
            payload = {
                "question": question,
                "top_k": top_k,
                "law_filter": law_filter
            }
            response = requests.post(f"{self.base_url}/question", json=payload)
            if response.status_code == 200:
                return response.json()
            return {"error": f"質問応答エラー: {response.status_code}"}
        except Exception as e:
            return {"error": f"接続エラー: {str(e)}"}

def initialize_session_state():
    """セッション状態の初期化"""
    if 'client' not in st.session_state:
        st.session_state.client = TaxLawClient()
    
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    if 'available_laws' not in st.session_state:
        st.session_state.available_laws = []

def check_backend_connection():
    """バックエンド接続確認"""
    if st.session_state.client.health_check():
        st.success("✅ バックエンドサーバーに接続済み")
        
        # 利用可能な法令を取得
        if not st.session_state.available_laws:
            st.session_state.available_laws = st.session_state.client.get_available_laws()
        
        return True
    else:
        st.error("❌ バックエンドサーバーに接続できません")
        st.info("バックエンドサーバー（http://localhost:8000）が起動していることを確認してください")
        return False

def display_search_result(result: Dict, index: int):
    """検索結果を表示"""
    with st.container():
        st.markdown(f"""
        <div class="search-result">
            <div class="law-name">{result['law_name']}</div>
            <div class="article-info">第{result['article_number']}条 {result['article_title']} 第{result['paragraph_number']}項</div>
            <div style="margin: 0.5rem 0;">
                <span class="similarity-score">類似度: {result['similarity']:.3f}</span>
            </div>
            <div style="margin-top: 1rem; line-height: 1.6;">
                {result['text']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 詳細情報の展開
        with st.expander(f"詳細情報 - {result['law_name']} 第{result['article_number']}条"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**法令名:** {result['law_name']}")
                st.write(f"**条文番号:** 第{result['article_number']}条")
                st.write(f"**項番号:** 第{result['paragraph_number']}項")
            with col2:
                st.write(f"**条文タイトル:** {result['article_title']}")
                st.write(f"**類似度スコア:** {result['similarity']:.6f}")
                st.write(f"**チャンクID:** {result['chunk_id']}")
            
            if result.get('url'):
                st.write(f"**法令URL:** [e-govで確認]({result['url']})")

def search_page():
    """検索ページ"""
    st.markdown('<div class="main-header">⚖️ 税法条文検索</div>', unsafe_allow_html=True)
    
    # 検索フォーム
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "検索キーワードを入力してください",
            placeholder="例: 法人税 税率, 消費税 軽減税率, 所得控除",
            help="法令条文の内容に関連するキーワードを入力してください"
        )
    
    with col2:
        top_k = st.selectbox("表示件数", [3, 5, 10, 20], index=1)
    
    # 法令フィルター
    if st.session_state.available_laws:
        selected_laws = st.multiselect(
            "検索対象法令を選択（空の場合は全法令を対象）",
            st.session_state.available_laws,
            help="特定の法令のみを検索対象にする場合は選択してください"
        )
    else:
        selected_laws = []
    
    # 検索実行
    if st.button("🔍 検索実行", type="primary") and search_query:
        with st.spinner("検索中..."):
            law_filter = selected_laws if selected_laws else None
            result = st.session_state.client.search(search_query, top_k, law_filter)
            
            if "error" in result:
                st.error(f"検索エラー: {result['error']}")
                return
            
            # 検索履歴に追加
            st.session_state.search_history.append({
                "timestamp": datetime.now(),
                "query": search_query,
                "results_count": len(result["results"])
            })
            
            # 結果表示
            st.success(f"✅ {len(result['results'])} 件の関連条文が見つかりました")
            
            if result["results"]:
                # 統計情報
                similarities = [r["similarity"] for r in result["results"]]
                laws_found = list(set(r["law_name"] for r in result["results"]))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("検索結果数", len(result["results"]))
                with col2:
                    st.metric("最高類似度", f"{max(similarities):.3f}")
                with col3:
                    st.metric("関連法令数", len(laws_found))
                
                # 結果一覧
                st.subheader("検索結果")
                for i, search_result in enumerate(result["results"]):
                    display_search_result(search_result, i)
                
                # 類似度分布チャート
                if len(similarities) > 1:
                    st.subheader("類似度分布")
                    fig = px.bar(
                        x=range(1, len(similarities) + 1),
                        y=similarities,
                        labels={"x": "検索結果順位", "y": "類似度スコア"},
                        title="検索結果の類似度分布"
                    )
                    st.plotly_chart(fig, use_container_width=True)

def question_page():
    """質問応答ページ"""
    st.markdown('<div class="main-header">💬 税法質問応答</div>', unsafe_allow_html=True)
    
    # 質問フォーム
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_area(
            "税法に関する質問を入力してください",
            placeholder="例: 法人税の税率は何％ですか？\n消費税の軽減税率の対象は何ですか？\n所得税の基礎控除額はいくらですか？",
            height=100,
            help="自然言語で質問してください。AIが関連する法令条文を参照して回答します。"
        )
    
    with col2:
        top_k = st.selectbox("参照条文数", [1, 2, 3, 5], index=2)
    
    # 法令フィルター
    if st.session_state.available_laws:
        selected_laws = st.multiselect(
            "参照対象法令を選択（空の場合は全法令を対象）",
            st.session_state.available_laws,
            key="qa_law_filter"
        )
    else:
        selected_laws = []
    
    # 質問実行
    if st.button("💡 質問する", type="primary") and question:
        with st.spinner("回答生成中..."):
            law_filter = selected_laws if selected_laws else None
            result = st.session_state.client.ask_question(question, top_k, law_filter)
            
            if "error" in result:
                st.error(f"質問応答エラー: {result['error']}")
                return
            
            # 回答表示
            st.markdown("""
            <div class="answer-box">
                <h3>🤖 AI回答</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(result["answer"])
            
            # 参照条文表示
            if result["relevant_chunks"]:
                st.subheader("📚 参照した法令条文")
                st.info(f"{len(result['relevant_chunks'])} 件の条文を参照して回答しました")
                
                for i, chunk in enumerate(result["relevant_chunks"]):
                    with st.expander(f"参照条文 {i+1}: {chunk['law_name']} 第{chunk['article_number']}条"):
                        display_search_result(chunk, i)

def analytics_page():
    """分析ページ"""
    st.markdown('<div class="main-header">📊 システム分析</div>', unsafe_allow_html=True)
    
    # システム情報取得
    system_info = st.session_state.client.get_system_info()
    
    if system_info:
        # 基本統計
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("総チャンク数", system_info.get("total_chunks", 0))
        with col2:
            st.metric("総法令数", system_info.get("total_laws", 0))
        with col3:
            if "index_info" in system_info:
                created_at = system_info["index_info"].get("created_at", "不明")
                st.metric("インデックス作成日", created_at[:10] if created_at != "不明" else "不明")
        
        # 法令別チャンク数
        if "available_laws" in system_info:
            st.subheader("法令別の登録状況")
            laws = system_info["available_laws"]
            
            # 簡易的な統計（実際にはバックエンドからより詳細な情報を取得することを推奨）
            law_data = pd.DataFrame({
                "法令名": laws,
                "状態": ["登録済み"] * len(laws)
            })
            
            st.dataframe(law_data, use_container_width=True)
        
        # インデックス情報
        if "index_info" in system_info:
            st.subheader("インデックス詳細情報")
            index_info = system_info["index_info"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.json({
                    "作成日時": index_info.get("created_at", "不明"),
                    "エンベディングモデル": index_info.get("embedding_model", "不明"),
                    "エンベディング次元": index_info.get("embedding_dimension", "不明")
                })
            
            with col2:
                if "processed_laws" in index_info:
                    st.write("**処理済み法令:**")
                    for law in index_info["processed_laws"]:
                        st.write(f"• {law}")
    
    # 検索履歴
    if st.session_state.search_history:
        st.subheader("検索履歴")
        
        history_df = pd.DataFrame(st.session_state.search_history)
        history_df["日時"] = history_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        st.dataframe(
            history_df[["日時", "query", "results_count"]].rename(columns={
                "query": "検索クエリ",
                "results_count": "結果数"
            }),
            use_container_width=True
        )
        
        # 検索履歴のクリア
        if st.button("🗑️ 検索履歴をクリア"):
            st.session_state.search_history = []
            st.success("検索履歴をクリアしました")
            st.experimental_rerun()

def main():
    """メイン関数"""
    initialize_session_state()
    
    # サイドバー
    with st.sidebar:
        st.title("税法検索システム")
        st.markdown("---")
        
        # 接続状態確認
        backend_connected = check_backend_connection()
        
        if backend_connected:
            # システム情報
            system_info = st.session_state.client.get_system_info()
            if system_info:
                st.metric("登録法令数", len(system_info.get("available_laws", [])))
                st.metric("総チャンク数", system_info.get("total_chunks", 0))
        
        st.markdown("---")
        
        # ページ選択
        page = st.selectbox(
            "ページを選択",
            ["🔍 条文検索", "💬 質問応答", "📊 システム分析"],
            disabled=not backend_connected
        )
        
        st.markdown("---")
        st.info("""
        **使い方:**
        1. 条文検索: キーワードで関連条文を検索
        2. 質問応答: 自然言語での質問に AI が回答
        3. システム分析: 登録データの統計情報
        """)
    
    # メインコンテンツ
    if backend_connected:
        if page == "🔍 条文検索":
            search_page()
        elif page == "💬 質問応答":
            question_page()
        elif page == "📊 システム分析":
            analytics_page()
    else:
        st.warning("バックエンドサーバーへの接続が必要です")

if __name__ == "__main__":
    main()
