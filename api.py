from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json
import openai
import os
from typing import List, Dict, Optional
import uvicorn

# リクエスト・レスポンスモデル
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    law_filter: Optional[List[str]] = None

class SearchResult(BaseModel):
    chunk_id: str
    law_name: str
    article_number: str
    article_title: str
    paragraph_number: int
    text: str
    similarity: float
    url: str

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_chunks: int
    query: str

class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    law_filter: Optional[List[str]] = None

class QuestionResponse(BaseModel):
    answer: str
    relevant_chunks: List[SearchResult]
    question: str

class LawSearchEngine:
    """法令検索エンジン"""
    
    def __init__(self, data_dir: str, openai_api_key: str):
        self.data_dir = data_dir
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.chunks = []
        self.embeddings = None
        self.metadata = {}
        self.index_info = {}
        
        self.load_data()
    
    def load_data(self):
        """ベクトル化データを読み込み"""
        try:
            # チャンクデータ
            with open(os.path.join(self.data_dir, "chunks.json"), "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            
            # エンベディング
            self.embeddings = np.load(os.path.join(self.data_dir, "embeddings.npy"))
            
            # メタデータ
            with open(os.path.join(self.data_dir, "metadata.json"), "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            
            # インデックス情報
            try:
                with open(os.path.join(self.data_dir, "index_info.json"), "r", encoding="utf-8") as f:
                    self.index_info = json.load(f)
            except:
                self.index_info = {}
            
            print(f"データ読み込み完了:")
            print(f"- チャンク数: {len(self.chunks)}")
            print(f"- エンベディング形状: {self.embeddings.shape}")
            print(f"- 法令数: {len(self.metadata)}")
            
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            raise
    
    def search_similar(self, query: str, top_k: int = 5, law_filter: Optional[List[str]] = None) -> List[Dict]:
        """類似度検索"""
        if self.embeddings is None:
            raise Exception("エンベディングデータが読み込まれていません")
        
        # クエリのエンベディング生成
        try:
            query_response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=query
            )
            query_embedding = np.array(query_response.data[0].embedding)
        except Exception as e:
            raise Exception(f"クエリのエンベディング生成に失敗: {e}")
        
        # 法令フィルタリング
        valid_indices = list(range(len(self.chunks)))
        if law_filter:
            valid_indices = [
                i for i, chunk in enumerate(self.chunks)
                if any(law in chunk['metadata']['law_name'] for law in law_filter)
            ]
        
        if not valid_indices:
            return []
        
        # 対象エンベディングを抽出
        filtered_embeddings = self.embeddings[valid_indices]
        
        # コサイン類似度計算
        similarities = np.dot(filtered_embeddings, query_embedding) / (
            np.linalg.norm(filtered_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # 上位k件を取得
        top_local_indices = np.argsort(similarities)[-top_k:][::-1]
        top_global_indices = [valid_indices[i] for i in top_local_indices]
        
        results = []
        for i, global_idx in enumerate(top_global_indices):
            chunk = self.chunks[global_idx]
            results.append({
                'chunk': chunk,
                'similarity': float(similarities[top_local_indices[i]])
            })
        
        return results
    
    def answer_question(self, question: str, top_k: int = 3, law_filter: Optional[List[str]] = None) -> Dict:
        """質問応答"""
        # 関連条文を検索
        relevant_chunks = self.search_similar(question, top_k, law_filter)
        
        if not relevant_chunks:
            return {
                'answer': "関連する法令条文が見つかりませんでした。",
                'relevant_chunks': []
            }
        
        # コンテキストを構築
        context = "\n\n".join([
            f"【関連度: {chunk['similarity']:.3f}】\n{chunk['chunk']['text']}"
            for chunk in relevant_chunks
        ])
        
        # プロンプトを構築
        prompt = f"""
あなたは税法の専門家です。以下の法令条文を参考に、質問に正確に回答してください。

関連する法令条文:
{context}

質問: {question}

回答の際の注意事項:
1. 法令条文の内容に基づいて回答してください
2. 該当する法令名と条文番号を明記してください
3. 複数の法令にまたがる場合は、それぞれを明記してください
4. 不明な点がある場合は、その旨を明記してください
5. 法的解釈や具体的な適用については、税理士等の専門家への相談を推奨してください

回答:
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "あなたは税法の専門家として、正確で有用な情報を提供します。法令条文に基づいた回答を心がけてください。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
        except Exception as e:
            answer = f"回答生成中にエラーが発生しました: {e}"
        
        return {
            'answer': answer,
            'relevant_chunks': relevant_chunks
        }
    
    def get_available_laws(self) -> List[str]:
        """利用可能な法令一覧を取得"""
        laws = set(chunk['metadata']['law_name'] for chunk in self.chunks)
        return sorted(list(laws))
    
    def get_system_info(self) -> Dict:
        """システム情報を取得"""
        return {
            'total_chunks': len(self.chunks),
            'total_laws': len(self.metadata),
            'available_laws': self.get_available_laws(),
            'index_info': self.index_info
        }

# FastAPIアプリケーション
app = FastAPI(
    title="税法検索API",
    description="e-gov法令データを使用した税法検索システム",
    version="1.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切なオリジンを指定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバル変数
search_engine = None

@app.on_event("startup")
async def startup_event():
    """起動時の初期化"""
    global search_engine
    
    data_dir = os.getenv("VECTORIZED_DATA_DIR", "./vectorized_data")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        raise Exception("OPENAI_API_KEY環境変数が設定されていません")
    
    if not os.path.exists(data_dir):
        raise Exception(f"ベクトル化データディレクトリが見つかりません: {data_dir}")
    
    search_engine = LawSearchEngine(data_dir, openai_api_key)
    print("検索エンジンの初期化が完了しました")

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {"message": "税法検索API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    if search_engine is None:
        raise HTTPException(status_code=503, detail="検索エンジンが初期化されていません")
    
    return {"status": "healthy", "total_chunks": len(search_engine.chunks)}

@app.get("/laws")
async def get_laws():
    """利用可能な法令一覧を取得"""
    if search_engine is None:
        raise HTTPException(status_code=503, detail="検索エンジンが初期化されていません")
    
    return {"laws": search_engine.get_available_laws()}

@app.get("/system-info")
async def get_system_info():
    """システム情報を取得"""
    if search_engine is None:
        raise HTTPException(status_code=503, detail="検索エンジンが初期化されていません")
    
    return search_engine.get_system_info()

@app.post("/search", response_model=SearchResponse)
async def search_laws(request: SearchRequest):
    """法令検索"""
    if search_engine is None:
        raise HTTPException(status_code=503, detail="検索エンジンが初期化されていません")
    
    try:
        results = search_engine.search_similar(
            request.query, 
            request.top_k, 
            request.law_filter
        )
        
        search_results = []
        for result in results:
            chunk = result['chunk']
            search_results.append(SearchResult(
                chunk_id=chunk['id'],
                law_name=chunk['metadata']['law_name'],
                article_number=chunk['metadata']['article_number'],
                article_title=chunk['metadata']['article_title'],
                paragraph_number=chunk['metadata']['paragraph_number'],
                text=chunk['text'],
                similarity=result['similarity'],
                url=chunk['metadata']['url']
            ))
        
        return SearchResponse(
            results=search_results,
            total_chunks=len(search_engine.chunks),
            query=request.query
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """質問応答"""
    if search_engine is None:
        raise HTTPException(status_code=503, detail="検索エンジンが初期化されていません")
    
    try:
        result = search_engine.answer_question(
            request.question,
            request.top_k,
            request.law_filter
        )
        
        relevant_chunks = []
        for chunk_result in result['relevant_chunks']:
            chunk = chunk_result['chunk']
            relevant_chunks.append(SearchResult(
                chunk_id=chunk['id'],
                law_name=chunk['metadata']['law_name'],
                article_number=chunk['metadata']['article_number'],
                article_title=chunk['metadata']['article_title'],
                paragraph_number=chunk['metadata']['paragraph_number'],
                text=chunk['text'],
                similarity=chunk_result['similarity'],
                url=chunk['metadata']['url']
            ))
        
        return QuestionResponse(
            answer=result['answer'],
            relevant_chunks=relevant_chunks,
            question=request.question
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )