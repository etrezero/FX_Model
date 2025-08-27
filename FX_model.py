# TensorFlow 로그 레벨 설정
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# TensorFlow 로그 추가 설정
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# 필수 라이브러리
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from flask import Flask

# 통계 및 머신러닝
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from scipy import stats
import xgboost as xgb

# 추가 라이브러리
import time
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import hashlib
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks

today = datetime.today().strftime("%Y-%m-%d")

# ============================================
# 캐시 디렉토리 설정
# ============================================
BASE_DIR = Path(__file__).resolve().parent if '__file__' in locals() else Path.cwd()
CACHE_DIR = BASE_DIR / "cache_fx"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# 급등/급락 디텍터 클래스 (DTW 추가)
# ============================================
class FXAnomalyDetector:
    """환율 이상 패턴 감지 - Isolation Forest & DTW 기반"""
    
    def __init__(self, data):
        self.data = data
        self.alerts = []
        self.bubble_periods = []
        self.anomaly_points = []
        self.dtw_anomalies = []
        self.dtw_distances = None
        self.dtw_timeline = None
        
    def detect_anomalies_isolation_forest(self, contamination=0.01):
        """Isolation Forest를 사용한 이상치 감지"""
        alerts = []
        
        # 환율 데이터와 변동률 준비
        df = self.data.copy()
        df['returns'] = df['USDKRW'].pct_change()
        df['log_rate'] = np.log(df['USDKRW'])
        df['vol_20'] = df['returns'].rolling(20).std()
        
        # 다차원 특성으로 이상치 감지
        features = ['USDKRW', 'returns']
        if 'USDKRW_vol' in df.columns:
            features.append('USDKRW_vol')
        if 'VIX' in df.columns:
            features.append('VIX')
            
        # NaN 제거
        df_clean = df[features].dropna()
        
        if len(df_clean) < 100:
            print("데이터 부족으로 Isolation Forest 스킵")
            return alerts
        
        # 데이터 정규화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean)
        
        # Isolation Forest 모델
        model = IsolationForest(
            contamination=contamination,
            max_samples='auto',
            random_state=42,
            n_estimators=100
        )
        
        # 이상치 예측
        df_clean['anomaly'] = model.fit_predict(X_scaled)
        df_clean['anomaly_score'] = model.score_samples(X_scaled)
        
        # 이상치 포인트 추출
        anomalies = df_clean[df_clean['anomaly'] == -1]
        
        for idx, row in anomalies.iterrows():
            # 이상치 유형 판단
            if 'returns' in df_clean.columns:
                ret_val = row.get('returns', 0)
                if ret_val > 0:
                    anomaly_type = '이상 급등'
                    severity = 'CRITICAL' if abs(ret_val) > 0.03 else 'HIGH'
                else:
                    anomaly_type = '이상 급락'
                    severity = 'CRITICAL' if abs(ret_val) > 0.03 else 'HIGH'
            else:
                anomaly_type = '이상 패턴'
                severity = 'MEDIUM'
            
            alerts.append({
                'date': idx,
                'type': anomaly_type,
                'severity': severity,
                'metric': 'Isolation Forest',
                'value': row['anomaly_score'],
                'rate': self.data.loc[idx, 'USDKRW'] if idx in self.data.index else row.get('USDKRW', 0)
            })
        
        self.alerts.extend(alerts)
        self.anomaly_points = anomalies.index.tolist()
        return alerts
    
    def detect_dtw_anomalies(self, window_size=20, stride=5, threshold_percentile=92):
        """Dynamic Time Warping 기반 이상치 감지"""
        
        def simple_dtw(s1, s2):
            """간단한 DTW 거리 계산"""
            n, m = len(s1), len(s2)
            dtw_matrix = np.zeros((n+1, m+1))
            dtw_matrix[0, :] = np.inf
            dtw_matrix[:, 0] = np.inf
            dtw_matrix[0, 0] = 0
            
            for i in range(1, n+1):
                for j in range(1, m+1):
                    cost = abs(s1[i-1] - s2[j-1])
                    dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # insertion
                                                   dtw_matrix[i, j-1],    # deletion
                                                   dtw_matrix[i-1, j-1])  # match
            
            return dtw_matrix[n, m]
        
        # 정규화된 환율 데이터
        rates = self.data['USDKRW'].values
        normalized_rates = (rates - np.mean(rates)) / (np.std(rates) + 1e-10)
        
        # 정상 패턴 정의 (처음 window_size * 2 구간의 평균 패턴)
        if len(normalized_rates) < window_size * 3:
            print("DTW를 위한 데이터가 부족합니다.")
            return []
        
        # 여러 정상 패턴 샘플링
        normal_patterns = []
        for i in range(0, min(100, len(normalized_rates) - window_size), 20):
            normal_patterns.append(normalized_rates[i:i+window_size])
        
        # 전체 시계열에 대해 슬라이딩 윈도우로 DTW 거리 계산
        dtw_distances = []
        timestamps = []
        
        for i in range(0, len(normalized_rates) - window_size + 1, stride):
            current_window = normalized_rates[i:i+window_size]
            
            # 각 정상 패턴과의 최소 DTW 거리 계산
            min_distance = min([simple_dtw(current_window, pattern) 
                              for pattern in normal_patterns])
            
            dtw_distances.append(min_distance)
            timestamps.append(self.data.index[i + window_size//2])
        
        dtw_distances = np.array(dtw_distances)
        self.dtw_distances = dtw_distances
        
        # 이상치 임계값 설정
        threshold = np.percentile(dtw_distances, threshold_percentile)
        
        # 이상치 탐지
        dtw_alerts = []
        for i, (dist, ts) in enumerate(zip(dtw_distances, timestamps)):
            if dist > threshold:
                # 실제 환율값 찾기
                actual_idx = self.data.index.get_loc(ts)
                rate_value = self.data['USDKRW'].iloc[actual_idx]
                
                # 이상치 유형 판단 (전후 비교)
                if actual_idx > 0 and actual_idx < len(self.data) - 1:
                    prev_rate = self.data['USDKRW'].iloc[actual_idx - 1]
                    next_rate = self.data['USDKRW'].iloc[actual_idx + 1]
                    
                    if rate_value > prev_rate and rate_value > next_rate:
                        anomaly_type = 'DTW 패턴 이상 (상승)'
                    elif rate_value < prev_rate and rate_value < next_rate:
                        anomaly_type = 'DTW 패턴 이상 (하락)'
                    else:
                        anomaly_type = 'DTW 패턴 변동'
                else:
                    anomaly_type = 'DTW 패턴 이상'
                
                severity = 'CRITICAL' if dist > np.percentile(dtw_distances, 99) else 'HIGH'
                
                dtw_alerts.append({
                    'date': ts,
                    'type': anomaly_type,
                    'severity': severity,
                    'metric': 'DTW',
                    'value': dist,
                    'rate': rate_value
                })
        
        self.dtw_anomalies = dtw_alerts
        self.alerts.extend(dtw_alerts)
        
        # DTW 거리 시계열 데이터 저장
        self.dtw_timeline = pd.Series(dtw_distances, index=timestamps)
        
        print(f"DTW 이상치 감지 완료: {len(dtw_alerts)}개 발견")
        return dtw_alerts
    
    def detect_volatility_spike(self, window=20, threshold=2):
        """변동성 스파이크 감지 (보조 지표)"""
        returns = self.data['USDKRW'].pct_change()
        rolling_vol = returns.rolling(window).std()
        vol_zscore = (rolling_vol - rolling_vol.mean()) / rolling_vol.std()
        
        spikes = vol_zscore[vol_zscore > threshold]
        
        alerts = []
        for date, zscore in spikes.items():
            alerts.append({
                'date': date,
                'type': '변동성 급증',
                'severity': 'HIGH' if zscore > 3 else 'MEDIUM',
                'metric': 'Z-score',
                'value': zscore,
                'rate': self.data.loc[date, 'USDKRW']
            })
        
        self.alerts.extend(alerts)
        return alerts
    
    def detect_psy_bubbles(self, max_lags=4, bootstrap_sims=50, alpha=0.95):
        """PSY 버블 탐지 (간소화 버전)"""
        try:
            y = np.log(self.data['USDKRW'].values)
            T = len(y)
            
            # 최소 윈도우 크기
            r0 = 0.01 + 1.8 / np.sqrt(T)
            w0 = max(int(np.floor(r0 * T)), 20)
            
            # BSADF 계산 (간소화)
            bsadf = np.full(T, np.nan)
            
            for r2 in range(w0, T):
                t_max = -np.inf
                for r1 in range(max(0, r2 - 100), r2 - w0 + 1):
                    sub = y[r1: r2 + 1]
                    if len(sub) < 10:
                        continue
                        
                    # 간단한 ADF 통계량 근사
                    dy = np.diff(sub)
                    y_lag = sub[:-1]
                    
                    if len(dy) > 0:
                        X = np.column_stack([np.ones(len(dy)), y_lag])
                        try:
                            beta = np.linalg.lstsq(X, dy, rcond=None)[0]
                            resid = dy - X @ beta
                            se = np.sqrt(np.sum(resid**2) / (len(dy) - 2))
                            t_stat = beta[1] / (se / np.sqrt(np.sum((y_lag - y_lag.mean())**2)) + 1e-10)
                            if t_stat > t_max:
                                t_max = t_stat
                        except:
                            continue
                            
                bsadf[r2] = t_max
            
            # 간단한 임계치 설정
            threshold = np.nanpercentile(bsadf[~np.isnan(bsadf)], alpha * 100) if np.sum(~np.isnan(bsadf)) > 0 else 1.645
            
            # 버블 기간 식별
            bubble_signal = (bsadf > threshold)
            bubble_periods = []
            
            in_bubble = False
            start_idx = None
            
            for i in range(len(bubble_signal)):
                if not np.isnan(bubble_signal[i]):
                    if bubble_signal[i] and not in_bubble:
                        in_bubble = True
                        start_idx = i
                    elif not bubble_signal[i] and in_bubble:
                        bubble_periods.append({
                            'start': self.data.index[start_idx],
                            'end': self.data.index[i-1],
                            'max_stat': np.nanmax(bsadf[start_idx:i])
                        })
                        in_bubble = False
            
            self.bubble_periods = bubble_periods
            return bubble_periods, bsadf
            
        except Exception as e:
            print(f"PSY 버블 탐지 오류: {e}")
            return [], np.array([])
    
    def get_current_risk_level(self):
        """현재 리스크 레벨 계산 (DTW 포함)"""
        if not len(self.data):
            return "NORMAL", 0
        
        # 최근 변동성
        recent_returns = self.data['USDKRW'].pct_change().iloc[-20:]
        current_vol = recent_returns.std()
        historical_vol = self.data['USDKRW'].pct_change().std()
        
        # 최근 추세
        ma5 = self.data['USDKRW'].rolling(5).mean().iloc[-1]
        ma20 = self.data['USDKRW'].rolling(20).mean().iloc[-1]
        trend_strength = abs(ma5 / ma20 - 1)
        
        # 리스크 점수 계산
        vol_score = min(current_vol / (historical_vol + 1e-10), 3) * 25
        trend_score = min(trend_strength * 100, 100) * 25
        
        # 최근 알림 횟수 (Isolation Forest)
        recent_if_alerts = [a for a in self.alerts 
                          if a['metric'] == 'Isolation Forest' and
                          (self.data.index[-1] - a['date']).days < 10]
        if_alert_score = min(len(recent_if_alerts) * 8, 25)
        
        # 최근 DTW 알림 횟수
        recent_dtw_alerts = [a for a in self.alerts 
                           if a['metric'] == 'DTW' and
                           (self.data.index[-1] - a['date']).days < 10]
        dtw_alert_score = min(len(recent_dtw_alerts) * 8, 25)
        
        total_score = vol_score + trend_score + if_alert_score + dtw_alert_score
        
        if total_score >= 70:
            return "CRITICAL", total_score
        elif total_score >= 50:
            return "HIGH", total_score
        elif total_score >= 30:
            return "MEDIUM", total_score
        else:
            return "NORMAL", total_score


# ============================================
# 실제 데이터 수집 클래스
# ============================================
class FXDataCollector:
    def __init__(self):
        self.data = {}
        self.cache_dir = CACHE_DIR
        
    def get_cache_path(self, cache_key):
        return self.cache_dir / f"{cache_key}.pkl"
    
    def load_cache(self, cache_key, cache_days=1):  # 기본 만료기간 1일
        cache_path = self.get_cache_path(cache_key)
        if not cache_path.exists():
            print(f"  ❌ 캐시 없음: {cache_key}")
            return None
        
        cache_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        cache_age = (datetime.now() - cache_mtime)
        
        if cache_age.days >= cache_days:
            print(f"  ⏰ 캐시 만료: {cache_key} (생성: {cache_mtime.strftime('%Y-%m-%d %H:%M')})")
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                print(f"  ✅ 캐시 유효: {cache_key} (나이: {cache_age.seconds//3600}시간 {(cache_age.seconds%3600)//60}분)")
                return data
        except:
            print(f"  ❌ 캐시 읽기 실패: {cache_key}")
            return None
    
    def save_cache(self, data, cache_key):
        """캐시 저장"""
        cache_path = self.get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"  💾 캐시 저장 완료: {cache_key} ({len(data)}개 행)")
            # 다른 캐시 파일들 정리 (선택사항)
            self.cleanup_old_caches()
        except Exception as e:
            print(f"  ❌ 캐시 저장 실패: {e}")
    
    def cleanup_old_caches(self, days=7):
        """오래된 캐시 파일 정리"""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_age.days > days:
                    cache_file.unlink()
                    print(f"  🗑️ 오래된 캐시 삭제: {cache_file.name}")
        except Exception as e:
            pass  # 캐시 정리 실패는 무시
    
    def fetch_single_ticker(self, ticker, start_date, end_date):
        """단일 티커 데이터 가져오기 - 실제 데이터만"""
        try:
            print(f"  {ticker} 실제 데이터 다운로드 중...")
            
            # yfinance로 실제 데이터 다운로드
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data is not None and not data.empty:
                if 'Close' in data.columns:
                    result = data['Close'].dropna()
                elif len(data.columns) == 1:
                    result = data.iloc[:, 0].dropna()
                else:
                    result = None
                
                if result is not None and len(result) > 50:
                    print(f"  ✅ {ticker} 완료 ({len(result)}개 데이터)")
                    return result
                    
        except Exception as e:
            print(f"  ❌ {ticker} 오류: {str(e)[:100]}")
        
        return None
    
    def fetch_data(self, start_date='2020-01-01', end_date=today, period_label='custom'):
        """실제 데이터만 수집 - 기간별 캐싱"""
        # 기간별 고유 캐시 키 생성
        cache_key = f"fx_real_{period_label}_{start_date[:10]}_{end_date[:10]}"
        
        # 캐시 확인 (만료기간 1일)
        cached_data = self.load_cache(cache_key, cache_days=1)
        if cached_data is not None:
            print(f"📂 캐시에서 {period_label} 데이터 로드...")
            print(f"  - 캐시된 데이터: {len(cached_data)}개 행")
            print(f"  - 최신 환율: {cached_data['USDKRW'].iloc[-1]:.2f} KRW/USD")
            return cached_data
        
        print("=== 실제 데이터 수집 시작 ===")
        
        # 필수 티커들
        tickers = {
            'USDKRW': 'KRW=X',      # 원/달러 환율
            'US_10Y': '^TNX',        # 미국 10년 국채
            'KOSPI': '^KS11',        # KOSPI
            'SP500': '^GSPC',        # S&P 500
            'OIL': 'CL=F',           # WTI 원유
            'VIX': '^VIX',           # VIX (공포지수)
            'DXY': 'DX-Y.NYB',       # 달러 인덱스
            'GOLD': 'GC=F'           # 금 선물
        }
        
        collected_data = {}
        
        # 각 티커 데이터 수집
        for name, ticker in tickers.items():
            time.sleep(1)  # API 제한 회피
            result = self.fetch_single_ticker(ticker, start_date, end_date)
            if result is not None:
                collected_data[name] = result
        
        # 필수 데이터 확인
        if 'USDKRW' not in collected_data or len(collected_data['USDKRW']) < 100:
            raise ValueError("❌ KRW=X 실제 환율 데이터를 가져올 수 없습니다. 네트워크 연결을 확인하세요.")
        
        # 공통 날짜 찾기
        common_dates = collected_data['USDKRW'].index
        for name, data in collected_data.items():
            if name != 'USDKRW':
                common_dates = common_dates.intersection(data.index)
        
        print(f"공통 날짜: {len(common_dates)}개")
        
        # 데이터프레임 구성
        processed_data = pd.DataFrame(index=common_dates)
        
        for name, data in collected_data.items():
            processed_data[name] = data.reindex(common_dates)
        
        # 결측치 처리 (앞/뒤 값으로 채우기)
        processed_data = processed_data.fillna(method='ffill').fillna(method='bfill')
        
        # 기술적 지표 계산
        if 'USDKRW' in processed_data.columns:
            processed_data['USDKRW_ret'] = processed_data['USDKRW'].pct_change()
            processed_data['USDKRW_MA5'] = processed_data['USDKRW'].rolling(5).mean()
            processed_data['USDKRW_MA20'] = processed_data['USDKRW'].rolling(20).mean()
            processed_data['USDKRW_vol'] = processed_data['USDKRW_ret'].rolling(20).std()
        
        # 다른 자산 수익률
        for col in ['KOSPI', 'SP500']:
            if col in processed_data.columns:
                processed_data[f'{col}_ret'] = processed_data[col].pct_change()
        
        # NaN 제거
        processed_data = processed_data.dropna()
        
        # 최소 데이터 확인
        if len(processed_data) < 100:
            raise ValueError(f"❌ 충분한 데이터가 없습니다. ({len(processed_data)}개 행)")
        
        # 캐시 저장
        self.save_cache(processed_data, cache_key)
        
        print(f"✅ 실제 데이터 처리 완료: {len(processed_data)}개")
        print(f"  - 기간: {period_label}")
        print(f"  - 시작일: {processed_data.index[0].strftime('%Y-%m-%d')}")
        print(f"  - 종료일: {processed_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"  - 현재 환율: {processed_data['USDKRW'].iloc[-1]:.2f} KRW/USD")
        print(f"💾 {period_label} 데이터 캐시 저장 (1일간 유효)")
        
        return processed_data


# ============================================
# 예측 모델 클래스
# ============================================
class FXPredictionModels:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy().sort_index()
        assert 'USDKRW' in self.data.columns, "USDKRW 열이 필요합니다."
        
        # 필요한 지표 확인/생성
        if 'USDKRW_ret' not in self.data.columns:
            self.data['USDKRW_ret'] = self.data['USDKRW'].pct_change()
        if 'USDKRW_MA5' not in self.data.columns:
            self.data['USDKRW_MA5'] = self.data['USDKRW'].rolling(5).mean()
        if 'USDKRW_MA20' not in self.data.columns:
            self.data['USDKRW_MA20'] = self.data['USDKRW'].rolling(20).mean()
        if 'USDKRW_vol' not in self.data.columns:
            self.data['USDKRW_vol'] = self.data['USDKRW_ret'].rolling(20).std()

        self.predictions = {}
        self.performance = {}
        self.future_predictions = {}
        self.fast_mode = True
        self.trained_models = {}

    def prepare_data(self, test_size: int = 126):
        df = self.data.copy()
        X, y = self._make_feature_df(df)
        
        if len(X) < test_size + 50:
            test_size = max(21, min(63, len(X) // 5))

        split = len(X) - test_size
        self.train_idx = X.index[:split]
        self.test_idx = X.index[split:]

        self.X_train, self.X_test = X.iloc[:split], X.iloc[split:]
        self.y_train, self.y_test = y.iloc[:split], y.iloc[split:]

        self.train_data = df.loc[:self.test_idx[0]].iloc[:-1]
        self.test_data = df.loc[self.test_idx.min():self.test_idx.max()]

    def _make_feature_df(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        d = df.copy()
        
        # 기본 특성
        d['ret1'] = d['USDKRW'].pct_change()
        for k in range(1, 6):
            d[f'ret1_lag{k}'] = d['ret1'].shift(k)
        d['ma5'] = d['USDKRW'].rolling(5).mean()
        d['ma20'] = d['USDKRW'].rolling(20).mean()
        d['ma_ratio'] = d['ma5'] / d['ma20'] - 1
        d['vol20'] = d['ret1'].rolling(20).std()

        # 외생변수
        exog_cols = []
        for c in ['KOSPI_ret', 'SP500_ret', 'US_10Y', 'OIL', 'VIX', 'DXY', 'GOLD']:
            if c in d.columns:
                exog_cols.append(c)

        feature_cols = [f'ret1_lag{k}' for k in range(1, 6)] + ['ma_ratio', 'vol20'] + exog_cols
        d['USDKRW_next'] = d['USDKRW'].shift(-1)

        X = d[feature_cols].dropna().copy()
        y = d['USDKRW_next'].reindex(X.index)

        return X, y

    def run_all_models(self):
        if self.test_data is None or self.X_test is None:
            raise RuntimeError("prepare_data()를 먼저 호출하세요.")

        runners = [
            self._run_random_walk,
            self._run_momentum,
        ]

        for fn in runners:
            name = fn.__name__.replace('_run_', '').replace('_', ' ').title().replace(' ', '_')
            try:
                pred = fn()
                if pred is None or len(pred) == 0:
                    continue
                pred = pred.reindex(self.test_data.index).dropna()
                self.predictions[name] = pred
                self._evaluate(name, pred)
            except Exception as e:
                print(f"❌ {name} 실행 오류: {e}")

    def _run_random_walk(self) -> pd.Series:
        y_test = self.test_data['USDKRW'].copy()
        y_prev_last_train = self.train_data['USDKRW'].iloc[-1]
        pred = y_test.shift(1)
        pred.iloc[0] = y_prev_last_train
        return pred

    def _run_momentum(self) -> pd.Series:
        alpha = 0.6
        d = self.data
        preds = []
        for t in self.test_data.index:
            prev = d.loc[:t, 'USDKRW'].iloc[-2]
            ma5 = d.loc[:t, 'USDKRW'].rolling(5).mean().iloc[-2]
            ma20 = d.loc[:t, 'USDKRW'].rolling(20).mean().iloc[-2]
            if np.isnan(ma5) or np.isnan(ma20) or ma20 == 0:
                preds.append(prev)
            else:
                drift = (ma5 / ma20 - 1.0)
                preds.append(prev * (1.0 + alpha * drift))
        return pd.Series(preds, index=self.test_data.index, name='Momentum')

    def _evaluate(self, model_name: str, pred: pd.Series):
        actual = self.test_data['USDKRW'].reindex(pred.index).dropna()
        pred = pred.reindex(actual.index).dropna()
        if len(pred) == 0:
            return

        try:
            mse = mean_squared_error(actual, pred)
            mae = mean_absolute_error(actual, pred)
            rmse = float(np.sqrt(mse))
            mape_vals = np.abs((actual - pred) / np.where(actual == 0, np.nan, actual))
            mape = float(np.nanmean(mape_vals) * 100)
            corr = float(np.corrcoef(actual, pred)[0, 1]) if len(pred) > 1 else 0.0
        except Exception:
            rmse = mae = mape = 9e9
            corr = 0.0

        self.performance[model_name] = {
            "RMSE": rmse, "MAE": mae, "MAPE": mape, "Correlation": corr
        }

    def predict_future(self):
        if not self.predictions:
            return {}
        
        current_rate = self.data['USDKRW'].iloc[-1]
        horizons = {'1M': 21, '3M': 63, '6M': 126, '12M': 252}
        
        self.future_predictions = {}
        
        # 학습 데이터 길이에 따른 신뢰도 가중치
        data_length = len(self.train_data)
        confidence_factor = min(data_length / 500, 1.0)  # 500일 이상이면 최대 신뢰도
        
        for model_name in self.predictions.keys():
            forecasts = {}
            
            if model_name == 'Random_Walk':
                # Random Walk: 변동성 기반 예측 범위 추가
                historical_vol = self.data['USDKRW'].pct_change().std()
                for period, days in horizons.items():
                    # 기간이 길수록 불확실성 증가
                    vol_adjustment = historical_vol * np.sqrt(days/252) * confidence_factor
                    # 평균 회귀 경향 반영
                    long_term_mean = self.train_data['USDKRW'].mean()
                    mean_reversion = 0.01 * (long_term_mean - current_rate) / current_rate
                    forecasts[period] = current_rate * (1 + mean_reversion * (days/252))
                    
            elif model_name == 'Momentum':
                # 데이터 기간에 따라 다른 모멘텀 윈도우 사용
                if data_length > 500:
                    lookback_short = 20
                    lookback_long = 60
                elif data_length > 250:
                    lookback_short = 15
                    lookback_long = 45
                else:
                    lookback_short = 10
                    lookback_long = 30
                    
                recent_trend = (self.data['USDKRW'].iloc[-lookback_short:].mean() / 
                              self.data['USDKRW'].iloc[-lookback_long:-lookback_short].mean() - 1)
                
                # 장기 추세 보정
                long_trend = (self.data['USDKRW'].iloc[-126:].mean() / 
                            self.data['USDKRW'].iloc[-252:-126].mean() - 1) if len(self.data) > 252 else 0
                
                for period, days in horizons.items():
                    # 단기와 장기 추세 결합
                    combined_trend = 0.7 * recent_trend + 0.3 * long_trend
                    annual_drift = combined_trend * (252 / lookback_short)
                    period_drift = annual_drift * (days / 252) * confidence_factor
                    # 평균 회귀 효과
                    mean_reversion_factor = 0.5 ** (days / 252)
                    forecasts[period] = current_rate * (1 + period_drift * mean_reversion_factor)
                    
            elif model_name == 'ARIMA' and 'ARIMA' in self.trained_models:
                model = self.trained_models['ARIMA']
                
                # 데이터 기간에 따른 예측 조정
                for period, days in horizons.items():
                    try:
                        fc = model.get_forecast(steps=days)
                        base_forecast = fc.predicted_mean.iloc[-1]
                        
                        # 학습 데이터 길이에 따른 신뢰도 조정
                        if data_length < 250:  # 1년 미만 데이터
                            # 현재 환율과 예측의 가중 평균
                            weight = 0.3 + 0.7 * confidence_factor
                            forecasts[period] = weight * base_forecast + (1-weight) * current_rate
                        else:
                            forecasts[period] = base_forecast
                            
                    except:
                        forecasts[period] = current_rate
            else:
                for period, days in horizons.items():
                    forecasts[period] = current_rate
            
            self.future_predictions[model_name] = forecasts
        
        # 앙상블 예측 (성능과 데이터 기간 가중)
        ensemble_forecasts = {}
        for period in horizons.keys():
            weighted_sum = 0
            total_weight = 0
            
            for model_name, forecast in self.future_predictions.items():
                if model_name in self.performance:
                    # RMSE 기반 가중치에 데이터 신뢰도 반영
                    base_weight = 1 / (self.performance[model_name]['RMSE'] + 1)
                    adjusted_weight = base_weight * (0.5 + 0.5 * confidence_factor)
                    
                    # 장기 예측일수록 ARIMA 가중치 증가
                    if model_name == 'ARIMA' and period in ['6M', '12M']:
                        adjusted_weight *= 1.5
                    
                    weighted_sum += forecast[period] * adjusted_weight
                    total_weight += adjusted_weight
            
            if total_weight > 0:
                ensemble_forecasts[period] = weighted_sum / total_weight
            else:
                ensemble_forecasts[period] = current_rate
        
        self.future_predictions['Ensemble'] = ensemble_forecasts
        
        # 예측 통계 추가
        self.future_predictions['stats'] = {
            'confidence': confidence_factor,
            'data_days': data_length,
            'base_rate': current_rate
        }
        
        return self.future_predictions


# ============================================
# Dash 앱 초기화
# ============================================
server = Flask(__name__)
app = dash.Dash(__name__, suppress_callback_exceptions=True, server=server)
app.title = "Real FX Model with DTW Detector"

# ============================================
# 레이아웃
# ============================================
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                background-attachment: fixed;
                min-height: 100vh;
                color: #ffffff;
            }
            
            .main-container {
                max-width: 1920px;
                margin: 0 auto;
                padding: 2rem;
            }
            
            .header {
                text-align: center;
                margin-bottom: 2rem;
                padding: 2rem;
                background: rgba(255, 255, 255, 0.03);
                border-radius: 24px;
                backdrop-filter: blur(20px);
            }
            
            .alert-banner {
                background: linear-gradient(135deg, rgba(248, 113, 113, 0.2) 0%, rgba(239, 68, 68, 0.2) 100%);
                border: 2px solid rgba(248, 113, 113, 0.5);
                border-radius: 16px;
                padding: 1.5rem;
                margin-bottom: 2rem;
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0% { opacity: 0.9; }
                50% { opacity: 1; }
                100% { opacity: 0.9; }
            }
            
            .risk-indicator {
                display: inline-flex;
                align-items: center;
                gap: 1rem;
                padding: 0.75rem 1.5rem;
                background: rgba(0, 0, 0, 0.3);
                border-radius: 100px;
                border: 2px solid;
                font-weight: 600;
            }
            
            .risk-NORMAL {
                border-color: #5eead4;
                color: #5eead4;
            }
            
            .risk-MEDIUM {
                border-color: #fbbf24;
                color: #fbbf24;
            }
            
            .risk-HIGH {
                border-color: #fb923c;
                color: #fb923c;
            }
            
            .risk-CRITICAL {
                border-color: #ef4444;
                color: #ef4444;
                animation: blink 1s infinite;
            }
            
            @keyframes blink {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            .control-panel {
                background: rgba(255, 255, 255, 0.04);
                border-radius: 16px;
                padding: 1.5rem;
                margin-bottom: 2rem;
                display: flex;
                align-items: center;
                gap: 2rem;
                flex-wrap: wrap;
            }
            
            .summary-cards {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1.5rem;
                margin-bottom: 2rem;
            }
            
            .summary-card {
                background: rgba(255, 255, 255, 0.04);
                border-radius: 16px;
                padding: 1.5rem;
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.08);
                text-align: center;
            }
            
            .alert-card {
                background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%);
                border: 2px solid rgba(239, 68, 68, 0.3);
            }
            
            .forecast-section {
                margin-bottom: 2rem;
            }
            
            .forecast-cards {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin-bottom: 1rem;
                padding: 1.5rem;
                background: linear-gradient(135deg, rgba(94, 234, 212, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
                border-radius: 16px;
                border: 2px solid rgba(94, 234, 212, 0.3);
            }
            
            .forecast-card {
                background: rgba(255, 255, 255, 0.06);
                border-radius: 12px;
                padding: 1rem;
                text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .methodology-card {
                background: rgba(255, 255, 255, 0.04);
                border-radius: 24px;
                padding: 1.5rem;
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.08);
                margin-top: 1rem;
            }
            
            .analysis-grid {
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 2rem;
                margin-bottom: 2rem;
            }

            .analysis-card {
                background: rgba(255, 255, 255, 0.04);
                border-radius: 24px;
                padding: 1.5rem;
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
                min-height: 450px;
                max-height: 600px;
                overflow-y: auto;
            }

            .full-width-card {
                grid-column: span 2;
            }

            
            .error-message {
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    dcc.Store(id='data-store', storage_type='memory'),
    dcc.Store(id='models-store', storage_type='memory'),
    dcc.Store(id='detector-store', storage_type='memory'),
    dcc.Interval(id='interval-component', interval=1000000, n_intervals=0),
    
    # 헤더
    html.Div([
        html.H1("Covenant FX Model", 
                style={
                    "font-size": "2.5rem",
                    "background": "linear-gradient(135deg, #5eead4 0%, #8b5cf6 100%)",
                    "-webkit-background-clip": "text",
                    "-webkit-text-fill-color": "transparent",
                    "margin-bottom": "0.5rem"
                }),
        html.P("실제 환율 데이터 기반 예측 & DTW + IF 이상치 탐지",
               style={"color": "rgba(255, 255, 255, 0.7)", "font-size": "1.1rem"})
    ], className="header"),
    
    # 리스크 알림 배너
    html.Div(id='alert-banner'),
    
    # 컨트롤 패널
    html.Div([
        html.Div([
            html.Label("기간 설정:", className="dropdown-label"),
            dcc.Dropdown(
                id='period-select',
                options=[
                    {'label': '1년', 'value': '1Y'},
                    {'label': '3년', 'value': '3Y'},
                    {'label': '5년', 'value': '5Y'}
                ],
                value='3Y',
                style={"width": "100px", "color": "black"}
            ),
        ]),
        
        html.Div([
            html.Label("테스트 기간:", className="dropdown-label"),
            dcc.Dropdown(
                id='test-period',
                options=[
                    {'label': '1개월', 'value': 21},
                    {'label': '3개월', 'value': 63},
                    {'label': '6개월', 'value': 126}
                ],
                value=63,
                style={"width": "100px", "color": "black"}
            ),
        ]),
        
        html.Button("🚀 실제 데이터 분석", 
                   id="run-analysis",
                   n_clicks=1,
                   style={
                       "padding": "0.75rem 2rem",
                       "background": "linear-gradient(135deg, #5eead4 0%, #8b5cf6 100%)",
                       "border": "none",
                       "borderRadius": "8px",
                       "color": "white",
                       "cursor": "pointer",
                       "fontSize": "16px",
                       "fontWeight": "500"
                   }),
        
        html.Div(id='status-message', 
                style={"color": "#5eead4", "marginLeft": "auto"})
    ], className="control-panel"),
    
    # 향후 예측 섹션
    html.Div(id='forecast-section', className="forecast-section"),
    
    # 요약 카드들
    html.Div(id='summary-cards', className="summary-cards"),
    
    # 메인 분석 그리드
    html.Div(id='analysis-content', className="analysis-grid"),
    
    # 로딩 인디케이터
    dcc.Loading(
        id="loading",
        type="default",
        color="#5eead4",
        children=html.Div(id="loading-output")
    )
    
], className="main-container")


# ============================================
# 콜백 함수
# ============================================
@app.callback(
    [Output('alert-banner', 'children'),
     Output('forecast-section', 'children'),
     Output('summary-cards', 'children'),
     Output('analysis-content', 'children'),
     Output('status-message', 'children'),
     Output('data-store', 'data'),
     Output('models-store', 'data'),
     Output('detector-store', 'data')],
    [Input('run-analysis', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    [State('period-select', 'value'),
     State('test-period', 'value')]
)
def run_analysis(n_clicks, n_intervals, period, test_period):
    
    if not n_clicks:
        return [], [], [], [], "실제 데이터를 분석하려면 버튼을 클릭하세요", {}, {}, {}
    
    try:
        # 기간 설정
        period_map = {
            '1Y': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
            '2Y': (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'),
            '3Y': (datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d'),
            '5Y': (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d')
        }
        start_date = period_map[period]
        
        # 실제 데이터 수집 (기간별 캐싱)
        collector = FXDataCollector()
        data = collector.fetch_data(start_date=start_date, end_date=today, period_label=period)
        
        if data is None or data.empty:
            error_msg = html.Div([
                html.H4("❌ 실제 데이터 로드 실패"),
                html.P("네트워크 연결을 확인하고 다시 시도하세요."),
                html.P("yfinance가 정상적으로 작동하는지 확인하세요.")
            ], className="error-message")
            return [], [], [error_msg], [], "❌ 실제 데이터 로드 실패", {}, {}, {}
        
        # 이상 탐지 실행 (DTW 추가)
        detector = FXAnomalyDetector(data)
        detector.detect_anomalies_isolation_forest(contamination=0.003)
        detector.detect_volatility_spike()
        detector.detect_dtw_anomalies(window_size=20, stride=5, threshold_percentile=92)  # DTW 추가!
        bubble_periods, bsadf = detector.detect_psy_bubbles()
        risk_level, risk_score = detector.get_current_risk_level()
        
        # 모델 실행
        models = FXPredictionModels(data)
        models.prepare_data(test_size=test_period)
        models.run_all_models()
        future_predictions = models.predict_future()
        
        # 현재 환율 정보
        current_rate = data['USDKRW'].iloc[-1]
        daily_change = data['USDKRW'].iloc[-1] - data['USDKRW'].iloc[-2]
        daily_pct = (daily_change / data['USDKRW'].iloc[-2]) * 100
        
        # 52주 최고/최저
        year_data = data['USDKRW'].iloc[-252:] if len(data) >= 252 else data['USDKRW']
        year_high = year_data.max()
        year_low = year_data.min()
        
        # 알림 배너 생성 (DTW 포함)
        alert_banner = []
        if risk_level in ['HIGH', 'CRITICAL']:
            recent_if_alerts = [a for a in detector.alerts 
                              if a['metric'] == 'Isolation Forest' and
                              (data.index[-1] - a['date']).days < 5]
            recent_dtw_alerts = [a for a in detector.alerts 
                               if a['metric'] == 'DTW' and
                               (data.index[-1] - a['date']).days < 5]
            
            alert_banner = html.Div([
                html.Div([
                    html.H3("⚠️ 환율 이상 신호 감지", style={'marginBottom': '1rem'}),
                    html.Div(
                        f"리스크 레벨: {risk_level} (점수: {risk_score:.1f}/100)",
                        className=f"risk-indicator risk-{risk_level}"
                    ),
                    html.Div([
                        html.P(f"최근 IF 알림: {len(recent_if_alerts)}건 | DTW 알림: {len(recent_dtw_alerts)}건", 
                              style={'marginTop': '1rem'}),
                        html.Ul([
                            html.Li(f"{a['date'].strftime('%Y-%m-%d')}: {a['type']} ({a['metric']})")
                            for a in (recent_if_alerts + recent_dtw_alerts)[:5]
                        ])
                    ]) if recent_if_alerts or recent_dtw_alerts else None
                ])
            ], className="alert-banner")
        
        # 예측 카드 및 방법론 설명 생성
        forecast_section = []
        
        # 예측 카드들
        if 'Ensemble' in future_predictions:
            forecast_cards_content = []
            ensemble_forecasts = future_predictions['Ensemble']
            
            periods = ['1M', '3M', '6M', '12M']
            period_labels = ['1개월', '3개월', '6개월', '12개월']
            
            for period, label in zip(periods, period_labels):
                forecast_rate = ensemble_forecasts[period]
                forecast_change = forecast_rate - current_rate
                forecast_pct = (forecast_change / current_rate) * 100
                
                change_color = '#5eead4' if forecast_change >= 0 else '#f87171'
                
                forecast_cards_content.append(
                    html.Div([
                        html.H4(f"🗓️ {label} 후"),
                        html.Div(f"₩{forecast_rate:,.0f}", style={'fontSize': '1.8rem', 'fontWeight': '700'}),
                        html.Div(
                            f"{forecast_change:+.0f} ({forecast_pct:+.1f}%)",
                            style={'color': change_color}
                        )
                    ], className="forecast-card")
                )
            
            forecast_section.append(
                html.Div(forecast_cards_content, className="forecast-cards")
            )
        
        # 예상환율 산출 방식 설명 (예측 카드 바로 아래)
        model_weights = {}
        if models.performance:
            total_weight = 0
            for model_name, perf in models.performance.items():
                weight = 1 / (perf['RMSE'] + 1)
                model_weights[model_name] = weight
                total_weight += weight
            if total_weight > 0:
                model_weights = {k: v/total_weight for k, v in model_weights.items()}
        
        confidence_factor = min(len(data) / 500, 1.0) if data is not None else 0
        
        methodology_card = html.Div([
            html.H3("📊 종합 예상환율 산출 방식"),
            html.Div([
                html.H4("1. 개별 모델 예측 방법", style={'color': '#5eead4', 'marginTop': '1rem'}),
                html.Ul([
                    html.Li(f"Random Walk: 현재 환율 {current_rate:.0f}원 + 과거 변동성 × √(기간/252)"),
                    html.Li(f"Momentum: 최근 20일 추세 {((data['USDKRW'].iloc[-20:].mean() / data['USDKRW'].iloc[-40:-20].mean() - 1) * 100):.2f}% 반영")
                ]),
                
                html.H4("2. 모델별 실제 가중치", style={'color': '#8b5cf6', 'marginTop': '1rem'}),
                html.Div([
                    html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("모델", style={'padding': '8px', 'borderBottom': '1px solid rgba(255,255,255,0.2)'}),
                                html.Th("RMSE", style={'padding': '8px', 'borderBottom': '1px solid rgba(255,255,255,0.2)'}),
                                html.Th("가중치", style={'padding': '8px', 'borderBottom': '1px solid rgba(255,255,255,0.2)'})
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(model_name.replace('_', ' '), style={'padding': '8px'}),
                                html.Td(f"{perf['RMSE']:.2f}", style={'padding': '8px'}),
                                html.Td(f"{model_weights.get(model_name, 0)*100:.1f}%", style={'padding': '8px'})
                            ]) for model_name, perf in models.performance.items()
                        ] if models.performance else [])
                    ], style={'width': '100%', 'marginTop': '0.5rem'})
                ]),
                
                html.H4("3. 실제 계산 예시 (1개월 예측)", style={'color': '#fbbf24', 'marginTop': '1rem'}),
                html.Div([
                    html.P(f"• Random Walk 예측: {future_predictions.get('Random_Walk', {}).get('1M', 0):.0f}원 × {model_weights.get('Random_Walk', 0)*100:.1f}%", 
                          style={'fontFamily': 'monospace', 'fontSize': '0.85rem'}),
                    html.P(f"• Momentum 예측: {future_predictions.get('Momentum', {}).get('1M', 0):.0f}원 × {model_weights.get('Momentum', 0)*100:.1f}%",
                          style={'fontFamily': 'monospace', 'fontSize': '0.85rem'}),
                    html.P(f"= 최종 예상: {future_predictions.get('Ensemble', {}).get('1M', 0):.0f}원",
                          style={'fontFamily': 'monospace', 'fontSize': '0.9rem', 'fontWeight': 'bold', 'marginTop': '0.5rem'})
                ] if future_predictions else [html.P("예측 데이터 없음")]),
                
                html.H4(f"4. 신뢰도 지표 (현재 {confidence_factor*100:.0f}%)", style={'color': '#f87171', 'marginTop': '1rem'}),
                html.Div([
                    html.Div([
                        html.Div(style={
                            'width': f"{confidence_factor*100}%",
                            'height': '20px',
                            'backgroundColor': '#5eead4' if confidence_factor > 0.8 else '#fbbf24' if confidence_factor > 0.5 else '#f87171',
                            'borderRadius': '4px',
                            'transition': 'width 0.5s'
                        })
                    ], style={
                        'width': '100%',
                        'height': '20px',
                        'backgroundColor': 'rgba(255,255,255,0.1)',
                        'borderRadius': '4px',
                        'marginBottom': '0.5rem'
                    }),
                    html.P(f"• 학습 데이터: {len(models.train_data) if models and hasattr(models, 'train_data') else 0}일 / 500일 (최적)"),
                    html.P(f"• 백테스팅: 최근 {test_period}일간 검증 완료"),
                    html.P(f"• 데이터 품질: {'높음' if confidence_factor > 0.8 else '보통' if confidence_factor > 0.5 else '낮음'}")
                ]),
                
                html.Div([
                    html.Strong("⚠️ 투자 경고", style={'color': '#ef4444', 'fontSize': '1.1rem'}),
                    html.Br(),
                    html.P("1. 이 예측은 과거 패턴의 통계적 분석일 뿐입니다"),
                    html.P("2. 실제 환율은 예측 불가능한 요인들(정치, 경제정책, 국제정세)에 영향받습니다"),
                    html.P("3. 절대 이 예측만으로 투자 결정을 내리지 마세요"),
                    html.P("4. 항상 전문가 상담과 추가 분석을 병행하세요")
                ], style={
                    'padding': '1rem',
                    'backgroundColor': 'rgba(239, 68, 68, 0.1)',
                    'borderRadius': '8px',
                    'marginTop': '1rem',
                    'border': '2px solid rgba(239, 68, 68, 0.3)'
                })
            ], style={'fontSize': '0.9rem', 'lineHeight': '1.6'})
        ], className="methodology-card")
        
        forecast_section.append(methodology_card)
        
        # 요약 카드 생성
        summary_cards = [
            html.Div([
                html.H4("현재 환율 (실제)"),
                html.H2(f"₩{current_rate:,.2f}"),
                html.P(f"{daily_change:+.2f} ({daily_pct:+.2f}%)",
                      style={'color': '#5eead4' if daily_change >= 0 else '#f87171'})
            ], className="summary-card"),
            
            html.Div([
                html.H4("52주 범위"),
                html.H2(f"L: {year_low:,.0f}"),
                html.H2(f"H: {year_high:,.0f}"),
                html.P(f"현재: {((current_rate-year_low)/(year_high-year_low)*100):.1f}% 위치",
                      style={'color': 'rgba(255,255,255,0.6)'})
            ], className="summary-card"),
            
            html.Div([
                html.H4("리스크 레벨"),
                html.H2(risk_level),
                html.P(f"점수: {risk_score:.1f}/100",
                      style={'color': 'rgba(255,255,255,0.6)'})
            ], className="summary-card alert-card" if risk_level in ['HIGH', 'CRITICAL'] else "summary-card"),
            
            html.Div([
                html.H4("데이터 기간"),
                html.H2(f"{len(data)}일"),
                html.P(f"{data.index[0].strftime('%Y-%m-%d')} ~ {data.index[-1].strftime('%Y-%m-%d')}",
                      style={'color': 'rgba(255,255,255,0.6)', 'fontSize': '0.8rem'})
            ], className="summary-card")
        ]
        
        # 분석 차트들 (페어별로 그래프와 설명 카드 조합)
        analysis_cards = []
        
        # 1. 환율 추이 페어
        fig_main = go.Figure()
        fig_main.add_trace(go.Scatter(
            x=data.index,
            y=data['USDKRW'],
            mode='lines',
            name='실제 환율',
            line=dict(color='white', width=2)
        ))
        
        if 'USDKRW_MA5' in data.columns:
            fig_main.add_trace(go.Scatter(
                x=data.index,
                y=data['USDKRW_MA5'],
                mode='lines',
                name='5일 이동평균',
                line=dict(color='#5eead4', width=1)
            ))
        
        if 'USDKRW_MA20' in data.columns:
            fig_main.add_trace(go.Scatter(
                x=data.index,
                y=data['USDKRW_MA20'],
                mode='lines',
                name='20일 이동평균',
                line=dict(color='#8b5cf6', width=1)
            ))
        
        fig_main.update_layout(
            title='실제 KRW/USD 환율 추이',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        analysis_cards.append(html.Div([
            html.H3("📈 환율 추이"),
            dcc.Graph(figure=fig_main)
        ], className="analysis-card"))
        
        analysis_cards.append(html.Div([
            html.H3("📖 환율 추이 분석 가이드"),
            html.Div([
                html.H4("이동평균선 해석", style={'color': '#5eead4', 'fontSize': '1rem'}),
                html.Ul([
                    html.Li("5일 MA (초록): 단기 추세 - 빠른 반응", style={'fontSize': '0.85rem'}),
                    html.Li("20일 MA (보라): 중기 추세 - 안정적 방향성", style={'fontSize': '0.85rem'}),
                ]),
                
                html.H4("골든크로스/데드크로스", style={'color': '#fbbf24', 'fontSize': '1rem', 'marginTop': '1rem'}),
                html.P("• 골든크로스: 5일 MA > 20일 MA 돌파 → 상승 신호 📈", style={'fontSize': '0.85rem'}),
                html.P("• 데드크로스: 5일 MA < 20일 MA 하락 → 하락 신호 📉", style={'fontSize': '0.85rem'}),
                
                html.H4("지지/저항선", style={'color': '#f87171', 'fontSize': '1rem', 'marginTop': '1rem'}),
                html.P("• 20일 MA는 주요 지지/저항선 역할", style={'fontSize': '0.85rem'}),
                html.P("• MA 위 = 상승 추세, MA 아래 = 하락 추세", style={'fontSize': '0.85rem'}),
                
                html.Div([
                    html.P("💡 활용법: MA 간격이 넓어지면 추세 강화, 좁아지면 추세 전환 가능", 
                        style={'fontSize': '0.85rem', 'fontWeight': 'bold'})
                ], style={'marginTop': '1rem', 'padding': '8px', 'backgroundColor': 'rgba(94, 234, 212, 0.1)', 'borderRadius': '6px'})
            ])
        ], className="analysis-card"))
        
        # 2. IF+DTW 이상치 탐지 페어
        if detector.alerts:
            alert_df = pd.DataFrame(detector.alerts)
            fig_alerts = go.Figure()
            
            # 전체 환율 라인
            fig_alerts.add_trace(go.Scatter(
                x=data.index,
                y=data['USDKRW'],
                mode='lines',
                name='환율',
                line=dict(color='rgba(255,255,255,0.7)', width=2)
            ))
            
            # DTW 거리 시계열 (보조 Y축)
            if hasattr(detector, 'dtw_timeline') and detector.dtw_timeline is not None and len(detector.dtw_timeline) > 0:
                # DTW 거리를 환율 스케일로 정규화
                dtw_normalized = detector.dtw_timeline.values
                dtw_min, dtw_max = dtw_normalized.min(), dtw_normalized.max()
                if dtw_max > dtw_min:
                    dtw_scaled = (dtw_normalized - dtw_min) / (dtw_max - dtw_min)
                    rate_range = data['USDKRW'].max() - data['USDKRW'].min()
                    dtw_scaled = dtw_scaled * rate_range * 0.2 + data['USDKRW'].min()
                    
                    fig_alerts.add_trace(go.Scatter(
                        x=detector.dtw_timeline.index,
                        y=dtw_scaled,
                        mode='lines',
                        name='DTW 거리 (정규화)',
                        line=dict(color='rgba(255, 195, 0, 0.6)', width=1, dash='dot'),
                        yaxis='y'
                    ))
            
            # Isolation Forest 이상치 포인트
            anomaly_data = alert_df[alert_df['metric'] == 'Isolation Forest']
            if not anomaly_data.empty:
                # 이상 급등 (빨간 원)
                surge_anomalies = anomaly_data[anomaly_data['type'] == '이상 급등']
                if not surge_anomalies.empty:
                    fig_alerts.add_trace(go.Scatter(
                        x=surge_anomalies['date'],
                        y=surge_anomalies['rate'],
                        mode='markers',
                        name='IF: 이상 급등',
                        marker=dict(
                            color='red',
                            size=12,
                            symbol='circle',
                            line=dict(color='white', width=2)
                        ),
                        text=[f"날짜: {d.strftime('%Y-%m-%d')}<br>환율: {r:,.0f}<br>IF 스코어: {v:.3f}" 
                              for d, r, v in zip(surge_anomalies['date'], 
                                                surge_anomalies['rate'], 
                                                surge_anomalies['value'])],
                        hovertemplate='%{text}<extra></extra>'
                    ))
                
                # 이상 급락 (파란 원)
                crash_anomalies = anomaly_data[anomaly_data['type'] == '이상 급락']
                if not crash_anomalies.empty:
                    fig_alerts.add_trace(go.Scatter(
                        x=crash_anomalies['date'],
                        y=crash_anomalies['rate'],
                        mode='markers',
                        name='IF: 이상 급락',
                        marker=dict(
                            color='blue',
                            size=12,
                            symbol='circle',
                            line=dict(color='white', width=2)
                        ),
                        text=[f"날짜: {d.strftime('%Y-%m-%d')}<br>환율: {r:,.0f}<br>IF 스코어: {v:.3f}" 
                              for d, r, v in zip(crash_anomalies['date'], 
                                                crash_anomalies['rate'], 
                                                crash_anomalies['value'])],
                        hovertemplate='%{text}<extra></extra>'
                    ))
            
            # DTW 이상치 포인트
            dtw_data = alert_df[alert_df['metric'] == 'DTW']
            if not dtw_data.empty:
                # DTW 상승 이상 (핑크 삼각형)
                dtw_surge = dtw_data[dtw_data['type'].str.contains('상승')]
                if not dtw_surge.empty:
                    fig_alerts.add_trace(go.Scatter(
                        x=dtw_surge['date'],
                        y=dtw_surge['rate'],
                        mode='markers',
                        name='DTW: 패턴 이상 (상승)',
                        marker=dict(
                            color='#FF69B4',
                            size=14,
                            symbol='triangle-up',
                            line=dict(color='white', width=2)
                        ),
                        text=[f"날짜: {d.strftime('%Y-%m-%d')}<br>환율: {r:,.0f}<br>DTW 거리: {v:.3f}" 
                              for d, r, v in zip(dtw_surge['date'], 
                                                dtw_surge['rate'], 
                                                dtw_surge['value'])],
                        hovertemplate='%{text}<extra></extra>'
                    ))
                
                # DTW 하락 이상 (시안 삼각형)
                dtw_crash = dtw_data[dtw_data['type'].str.contains('하락')]
                if not dtw_crash.empty:
                    fig_alerts.add_trace(go.Scatter(
                        x=dtw_crash['date'],
                        y=dtw_crash['rate'],
                        mode='markers',
                        name='DTW: 패턴 이상 (하락)',
                        marker=dict(
                            color='#00FFFF',
                            size=14,
                            symbol='triangle-down',
                            line=dict(color='white', width=2)
                        ),
                        text=[f"날짜: {d.strftime('%Y-%m-%d')}<br>환율: {r:,.0f}<br>DTW 거리: {v:.3f}" 
                              for d, r, v in zip(dtw_crash['date'], 
                                                dtw_crash['rate'], 
                                                dtw_crash['value'])],
                        hovertemplate='%{text}<extra></extra>'
                    ))
                
                # DTW 변동 이상 (초록 다이아몬드)
                dtw_other = dtw_data[dtw_data['type'].str.contains('변동')]
                if not dtw_other.empty:
                    fig_alerts.add_trace(go.Scatter(
                        x=dtw_other['date'],
                        y=dtw_other['rate'],
                        mode='markers',
                        name='DTW: 패턴 변동',
                        marker=dict(
                            color='#32CD32',
                            size=12,
                            symbol='diamond',
                            line=dict(color='white', width=1)
                        ),
                        text=[f"날짜: {d.strftime('%Y-%m-%d')}<br>환율: {r:,.0f}<br>DTW 거리: {v:.3f}" 
                              for d, r, v in zip(dtw_other['date'], 
                                                dtw_other['rate'], 
                                                dtw_other['value'])],
                        hovertemplate='%{text}<extra></extra>'
                    ))
            
            # 범례와 타이틀 설정
            fig_alerts.update_layout(
                title='Isolation Forest + DTW 이상치 탐지',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=450,
                hovermode='closest',
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10)
                ),
                margin=dict(b=100)
            )
            
            # 이상치 개수 표시
            if_count = len(anomaly_data) if not anomaly_data.empty else 0
            dtw_count = len(dtw_data) if not dtw_data.empty else 0
            
            # 두 개의 annotation 추가
            fig_alerts.add_annotation(
                text=f"IF 이상치: {if_count}개",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=14, color='#FF6B6B'),
                bgcolor='rgba(0,0,0,0.5)',
                borderpad=10
            )
            
            fig_alerts.add_annotation(
                text=f"DTW 이상치: {dtw_count}개",
                xref="paper", yref="paper",
                x=0.02, y=0.90,
                showarrow=False,
                font=dict(size=14, color='#FFC300'),
                bgcolor='rgba(0,0,0,0.5)',
                borderpad=10
            )
            
            analysis_cards.append(html.Div([
                html.H3("🎯 Isolation Forest + DTW 이상치 탐지"),
                dcc.Graph(figure=fig_alerts)
            ], className="analysis-card"))
            
            analysis_cards.append(html.Div([
                html.H3("📖 이상치 탐지 방법론"),
                html.Div([
                    html.H4("Isolation Forest (IF)", style={'color': '#5eead4', 'fontSize': '1rem'}),
                    html.Div([
                        html.P("• 다차원 데이터에서 비정상 패턴 감지", style={'fontSize': '0.85rem'}),
                        html.P("• 고립도가 높을수록 이상치 가능성 ↑", style={'fontSize': '0.85rem'}),
                        html.H5("🔴 빨간 원: 이상 급등", style={'fontSize': '0.85rem', 'color': '#ff6b6b'}),
                        html.H5("🔵 파란 원: 이상 급락", style={'fontSize': '0.85rem', 'color': '#4dabf7'})
                    ], style={'marginBottom': '10px', 'padding': '8px', 'backgroundColor': 'rgba(94, 234, 212, 0.05)', 'borderRadius': '6px'}),
                    
                    html.H4("Dynamic Time Warping (DTW)", style={'color': '#fbbf24', 'fontSize': '1rem', 'marginTop': '1rem'}),
                    html.Div([
                        html.P("• 시계열 패턴의 유사성 측정", style={'fontSize': '0.85rem'}),
                        html.P("• 정상 패턴과의 거리로 이상 감지", style={'fontSize': '0.85rem'}),
                        html.H5("🔺 분홍 삼각형: 패턴 이상(상승)", style={'fontSize': '0.85rem', 'color': '#ff69b4'}),
                        html.H5("🔻 청록 삼각형: 패턴 이상(하락)", style={'fontSize': '0.85rem', 'color': '#00ffff'}),
                        html.H5("💎 초록 다이아몬드: 패턴 변동", style={'fontSize': '0.85rem', 'color': '#32cd32'})
                    ], style={'marginBottom': '10px', 'padding': '8px', 'backgroundColor': 'rgba(139, 92, 246, 0.05)', 'borderRadius': '6px'}),
                    
                    html.H4("활용 전략", style={'color': '#f87171', 'fontSize': '1rem', 'marginTop': '1rem'}),
                    html.P("• IF+DTW 중복 신호 = 매우 강한 이상 신호", style={'fontSize': '0.85rem', 'fontWeight': 'bold'}),
                    html.P("• 이상 급등 후 → 조정 대비", style={'fontSize': '0.85rem'}),
                    html.P("• 이상 급락 후 → 반등 대비", style={'fontSize': '0.85rem'}),
                    
                    html.Div([
                        html.P("⚠️ 이상치 ≠ 즉시 반전. 추세와 함께 종합 판단 필요", 
                            style={'fontSize': '0.85rem', 'color': '#ef4444', 'fontWeight': 'bold'})
                    ], style={'marginTop': '1rem', 'padding': '8px', 'backgroundColor': 'rgba(239, 68, 68, 0.1)', 'borderRadius': '6px'})
                ])
            ], className="analysis-card"))
        
        # 3. PSY 버블 페어
        if len(bsadf) > 0:
            fig_bubble = go.Figure()
            
            fig_bubble.add_trace(go.Scatter(
                x=data.index[-len(bsadf):],
                y=bsadf,
                mode='lines',
                name='BSADF 통계량',
                line=dict(color='#5eead4', width=2)
            ))
            
            threshold = np.nanpercentile(bsadf[~np.isnan(bsadf)], 95) if np.sum(~np.isnan(bsadf)) > 0 else 1.645
            fig_bubble.add_hline(y=threshold, line_dash="dash", 
                               line_color="red", 
                               annotation_text="95% 임계치")
            
            for period in bubble_periods:
                fig_bubble.add_vrect(
                    x0=period['start'], x1=period['end'],
                    fillcolor="red", opacity=0.2,
                    layer="below", line_width=0
                )
            
            fig_bubble.update_layout(
                title='PSY 버블 탐지',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            
            analysis_cards.append(html.Div([
                html.H3("🫧 버블 탐지"),
                dcc.Graph(figure=fig_bubble)
            ], className="analysis-card"))
            
            analysis_cards.append(html.Div([
                html.H3("📖 PSY 버블 탐지 이론"),
                html.Div([
                    html.H4("작동 원리", style={'color': '#5eead4', 'fontSize': '1rem'}),
                    html.P("Phillips-Shi-Yu (2015) 방법론", style={'fontSize': '0.85rem', 'fontStyle': 'italic'}),
                    html.Ul([
                        html.Li("환율의 폭발적 상승 행동 감지", style={'fontSize': '0.85rem'}),
                        html.Li("정상: 랜덤워크 / 버블: 지수적 증가", style={'fontSize': '0.85rem'}),
                        html.Li("BSADF 통계량으로 버블 판단", style={'fontSize': '0.85rem'})
                    ]),
                    
                    html.H4("BSADF 해석", style={'color': '#fbbf24', 'fontSize': '1rem', 'marginTop': '1rem'}),
                    html.Div([
                        html.H5("📊 통계량 > 95% 임계치", style={'fontSize': '0.85rem', 'color': '#ef4444'}),
                        html.P("• 버블 신호 감지 🔴", style={'fontSize': '0.8rem', 'marginLeft': '15px'}),
                        html.P("• 투기적 거래 증가", style={'fontSize': '0.8rem', 'marginLeft': '15px'}),
                        html.P("• 급락 위험 증가", style={'fontSize': '0.8rem', 'marginLeft': '15px'}),
                        
                        html.H5("📊 통계량 < 95% 임계치", style={'fontSize': '0.85rem', 'color': '#5eead4', 'marginTop': '10px'}),
                        html.P("• 정상 상태 ⚪", style={'fontSize': '0.8rem', 'marginLeft': '15px'}),
                        html.P("• 펀더멘털 기반 움직임", style={'fontSize': '0.8rem', 'marginLeft': '15px'})
                    ], style={'padding': '8px', 'backgroundColor': 'rgba(139, 92, 246, 0.05)', 'borderRadius': '6px'}),
                    
                    html.H4("투자 시사점", style={'color': '#f87171', 'fontSize': '1rem', 'marginTop': '1rem'}),
                    html.P("✅ 버블 초기: 추세 추종 가능", style={'fontSize': '0.85rem'}),
                    html.P("⚠️ 버블 지속: 포지션 축소", style={'fontSize': '0.85rem'}),
                    html.P("🚨 버블 후기: 즉시 청산 고려", style={'fontSize': '0.85rem'}),
                    
                    html.Div([
                        html.P("💡 한계: 버블 종료 시점 예측 어려움. 다른 지표와 병행 필수", 
                            style={'fontSize': '0.85rem', 'fontWeight': 'bold'})
                    ], style={'marginTop': '1rem', 'padding': '8px', 'backgroundColor': 'rgba(94, 234, 212, 0.1)', 'borderRadius': '6px'})
                ])
            ], className="analysis-card"))
        
        # 4. 예측 모델 페어
        if models.predictions:
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=models.test_data.index,
                y=models.test_data['USDKRW'].values,
                mode='lines',
                name='실제 환율',
                line=dict(color='white', width=3)
            ))
            
            colors = ['#5eead4', '#8b5cf6', '#fbbf24', '#f87171']
            for i, (model_name, pred) in enumerate(models.predictions.items()):
                fig_pred.add_trace(go.Scatter(
                    x=pred.index,
                    y=pred.values,
                    mode='lines',
                    name=model_name.replace('_', ' '),
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                ))
            
            fig_pred.update_layout(
                title='Technical 모델',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            
            analysis_cards.append(html.Div([
                html.H3("🤖 Technical 모델"),
                dcc.Graph(figure=fig_pred)
            ], className="analysis-card"))
            
            analysis_cards.append(html.Div([
                html.H3("📖 예측 모델 설명"),
                html.Div([
                    html.H4("Random Walk", style={'color': '#5eead4', 'fontSize': '1rem'}),
                    html.P("• 미래 = 현재 + 랜덤 변동", style={'fontSize': '0.85rem'}),
                    html.P("• 효율적 시장 가설 기반", style={'fontSize': '0.85rem'}),
                    html.P("• 단기 예측에 효과적", style={'fontSize': '0.85rem'}),
                    
                    html.H4("Momentum", style={'color': '#8b5cf6', 'fontSize': '1rem', 'marginTop': '1rem'}),
                    html.P("• 최근 추세 지속 가정", style={'fontSize': '0.85rem'}),
                    html.P("• MA 비율로 추세 강도 측정", style={'fontSize': '0.85rem'}),
                    html.P("• 트렌드 시장에서 우수", style={'fontSize': '0.85rem'}),
                    
                    html.H4("성능 평가 지표", style={'color': '#fbbf24', 'fontSize': '1rem', 'marginTop': '1rem'}),
                    html.Table([
                        html.Tbody([
                            html.Tr([
                                html.Td("RMSE", style={'fontSize': '0.8rem', 'padding': '3px', 'fontWeight': 'bold'}),
                                html.Td("오차의 제곱근 평균 (낮을수록 좋음)", style={'fontSize': '0.8rem', 'padding': '3px'})
                            ]),
                            html.Tr([
                                html.Td("MAE", style={'fontSize': '0.8rem', 'padding': '3px', 'fontWeight': 'bold'}),
                                html.Td("절대 오차 평균 (낮을수록 좋음)", style={'fontSize': '0.8rem', 'padding': '3px'})
                            ]),
                            html.Tr([
                                html.Td("MAPE", style={'fontSize': '0.8rem', 'padding': '3px', 'fontWeight': 'bold'}),
                                html.Td("평균 절대 퍼센트 오차 (%)", style={'fontSize': '0.8rem', 'padding': '3px'})
                            ]),
                            html.Tr([
                                html.Td("Corr", style={'fontSize': '0.8rem', 'padding': '3px', 'fontWeight': 'bold'}),
                                html.Td("예측과 실제의 상관관계", style={'fontSize': '0.8rem', 'padding': '3px'})
                            ])
                        ])
                    ], style={'width': '100%'}),
                    
                    html.Div([
                        html.P("💡 앙상블: 각 모델의 장점을 결합한 최종 예측", 
                            style={'fontSize': '0.85rem', 'fontWeight': 'bold'})
                    ], style={'marginTop': '1rem', 'padding': '8px', 'backgroundColor': 'rgba(94, 234, 212, 0.1)', 'borderRadius': '6px'})
                ])
            ], className="analysis-card"))
        
        # 5. 한국 특화 환율 인덱스 (전체 너비)
        # DXY-USDKRW 관계 분석 코드... (이전과 동일)
        # 한국 환율 인덱스 계산
        krw_index = np.zeros(len(data))  # 0으로 초기화 (NaN 대신)

        # 구성 요소별 가중치
        weights = {
            'dxy': 0.30,      # 달러인덱스
            'kospi': -0.20,   # KOSPI (역상관)
            'vix': 0.15,      # 변동성 지수
            'oil': 0.10,      # 원유 (수입 물가)
            'gold': -0.10,    # 금 (안전자산)
            'rate_diff': 0.15 # 금리차
        }

        # 각 요소 정규화 및 지수 계산
        components = {}

        # DXY
        if 'DXY' in data.columns:
            dxy_std = data['DXY'].std()
            if dxy_std > 0:
                components['dxy'] = ((data['DXY'] - data['DXY'].mean()) / dxy_std).fillna(0).values
            else:
                components['dxy'] = np.zeros(len(data))

        # KOSPI (역상관이므로 음수)
        if 'KOSPI' in data.columns:
            kospi_std = data['KOSPI'].std()
            if kospi_std > 0:
                components['kospi'] = -((data['KOSPI'] - data['KOSPI'].mean()) / kospi_std).fillna(0).values
            else:
                components['kospi'] = np.zeros(len(data))

        # VIX
        if 'VIX' in data.columns:
            vix_std = data['VIX'].std()
            if vix_std > 0:
                components['vix'] = ((data['VIX'] - data['VIX'].mean()) / vix_std).fillna(0).values
            else:
                components['vix'] = np.zeros(len(data))

        # OIL
        if 'OIL' in data.columns:
            oil_std = data['OIL'].std()
            if oil_std > 0:
                components['oil'] = ((data['OIL'] - data['OIL'].mean()) / oil_std).fillna(0).values
            else:
                components['oil'] = np.zeros(len(data))

        # GOLD
        if 'GOLD' in data.columns:
            gold_std = data['GOLD'].std()
            if gold_std > 0:
                components['gold'] = -((data['GOLD'] - data['GOLD'].mean()) / gold_std).fillna(0).values
            else:
                components['gold'] = np.zeros(len(data))

        # 금리차 (미국 10년물 기준)
        if 'US_10Y' in data.columns:
            kr_base_rate = 3.5  # 한국 기준금리 추정치
            components['rate_diff'] = ((data['US_10Y'] - kr_base_rate) / 2.0).fillna(0).values

        # 종합 인덱스 계산
        for component, weight in weights.items():
            if component in components:
                krw_index += components[component] * weight

        # Series로 변환 (인덱스 포함)
        krw_index = pd.Series(krw_index, index=data.index)

        # 인덱스를 100 기준으로 정규화
        krw_index_normalized = 100 + krw_index * 10
        
        # 통계 계산
        index_mean = krw_index_normalized.mean()
        index_std = krw_index_normalized.std()
        current_index = krw_index_normalized.iloc[-1]
        z_score = (current_index - index_mean) / index_std
        
        # 예측 신호 생성
        if z_score > 2:
            signal = "매우 과매수 → 하락 예상 🔴"
            signal_color = "#ef4444"
        elif z_score > 1:
            signal = "과매수 → 조정 가능 🟡"
            signal_color = "#fbbf24"
        elif z_score < -2:
            signal = "매우 과매도 → 상승 예상 🟢"
            signal_color = "#5eead4"
        elif z_score < -1:
            signal = "과매도 → 반등 가능 🟡"
            signal_color = "#fbbf24"
        else:
            signal = "중립 구간 → 관망 ⚪"
            signal_color = "#9ca3af"
        
        # 차트 생성
        fig_index = go.Figure()
        
        # 환율 (보조 Y축) - 색상을 화이트로 변경
        fig_index.add_trace(go.Scatter(
            x=data.index,
            y=data['USDKRW'],
            name='KRW/USD',
            line=dict(color='white', width=1.5),  # rgba(255,255,255,0.5) → white로 변경, 두께도 1.5로 증가
            yaxis='y2'
        ))

        # KRW 인덱스
        fig_index.add_trace(go.Scatter(
            x=krw_index_normalized.index,
            y=krw_index_normalized,
            name='KRW 인덱스',
            line=dict(color='#5eead4', width=2)
        ))
        
        # 평균선
        fig_index.add_hline(
            y=index_mean,
            line_dash="dash",
            line_color="white",
            annotation_text="평균"
        )
        
        # 1σ 밴드
        fig_index.add_hline(
            y=index_mean + index_std,
            line_dash="dot",
            line_color="yellow",
            annotation_text="+1σ 과매수"
        )
        fig_index.add_hline(
            y=index_mean - index_std,
            line_dash="dot",
            line_color="yellow",
            annotation_text="-1σ 과매도"
        )
        
        # 2σ 밴드
        fig_index.add_hline(
            y=index_mean + 2*index_std,
            line_dash="dot",
            line_color="red",
            annotation_text="+2σ 강한 과매수"
        )
        fig_index.add_hline(
            y=index_mean - 2*index_std,
            line_dash="dot",
            line_color="green",
            annotation_text="-2σ 강한 과매도"
        )
        
        # 현재 위치 표시
        fig_index.add_trace(go.Scatter(
            x=[krw_index_normalized.index[-1]],
            y=[current_index],
            mode='markers',
            name='현재',
            marker=dict(
                color=signal_color,
                size=15,
                symbol='star',
                line=dict(color='white', width=2)
            ),
            showlegend=False
        ))
        
        fig_index.update_layout(
            title='KRW Index(Macro embedded)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=450,
            yaxis=dict(
                title='KRW Index',
                side='left'
            ),
            yaxis2=dict(
                title='KRW/USD',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # 한국 특화 인덱스 카드 (전체 너비)
        index_card = html.Div([
            html.Div([
                html.H3("🎯 KRW Index(Macro embedded)"),
                html.Div([
                    html.Div([
                        html.H4(f"현재 지수: {current_index:.1f}", style={'display': 'inline-block'}),
                        html.Span(f" (Z-score: {z_score:+.2f})", style={'color': 'rgba(255,255,255,0.6)', 'marginLeft': '10px'})
                    ]),
                    html.Div([
                        html.H3(signal, style={'color': signal_color, 'marginTop': '10px'})
                    ])
                ], style={'padding': '1rem', 'backgroundColor': 'rgba(0,0,0,0.3)', 'borderRadius': '8px', 'marginBottom': '1rem'}),
                
                dcc.Graph(figure=fig_index),
                
                html.Div([
                    html.H4("인덱스 구성 요소", style={'marginTop': '1rem', 'marginBottom': '0.5rem'}),
                    html.Table([
                        html.Tbody([
                            html.Tr([
                                html.Td("달러인덱스 (DXY)", style={'padding': '4px'}),
                                html.Td("30%", style={'padding': '4px', 'textAlign': 'right'}),
                                html.Td("달러 강세 → 원화 약세", style={'padding': '4px', 'color': 'rgba(255,255,255,0.6)'})
                            ]),
                            html.Tr([
                                html.Td("KOSPI", style={'padding': '4px'}),
                                html.Td("-20%", style={'padding': '4px', 'textAlign': 'right'}),
                                html.Td("주가 상승 → 원화 강세", style={'padding': '4px', 'color': 'rgba(255,255,255,0.6)'})
                            ]),
                            html.Tr([
                                html.Td("VIX (공포지수)", style={'padding': '4px'}),
                                html.Td("15%", style={'padding': '4px', 'textAlign': 'right'}),
                                html.Td("불안 증가 → 원화 약세", style={'padding': '4px', 'color': 'rgba(255,255,255,0.6)'})
                            ]),
                            html.Tr([
                                html.Td("원유 가격", style={'padding': '4px'}),
                                html.Td("10%", style={'padding': '4px', 'textAlign': 'right'}),
                                html.Td("유가 상승 → 수입물가↑ → 원화 약세", style={'padding': '4px', 'color': 'rgba(255,255,255,0.6)'})
                            ]),
                            html.Tr([
                                html.Td("금 가격", style={'padding': '4px'}),
                                html.Td("-10%", style={'padding': '4px', 'textAlign': 'right'}),
                                html.Td("안전자산 선호 → 원화 약세", style={'padding': '4px', 'color': 'rgba(255,255,255,0.6)'})
                            ]),
                            html.Tr([
                                html.Td("금리차 (美-韓)", style={'padding': '4px'}),
                                html.Td("15%", style={'padding': '4px', 'textAlign': 'right'}),
                                html.Td("미국 금리↑ → 달러 선호 → 원화 약세", style={'padding': '4px', 'color': 'rgba(255,255,255,0.6)'})
                            ])
                        ])
                    ], style={'fontSize': '0.85rem', 'width': '100%'}),
                    
                    html.Div([
                        html.H4("신호 해석", style={'marginTop': '1rem'}),
                        html.P("• +2σ 이상: 원화 매우 약세, 반전 가능성 높음"),
                        html.P("• +1σ ~ +2σ: 원화 약세, 조정 가능"),
                        html.P("• -1σ ~ +1σ: 정상 범위"),
                        html.P("• -2σ ~ -1σ: 원화 강세, 반등 가능"),
                        html.P("• -2σ 이하: 원화 매우 강세, 반전 가능성 높음")
                    ], style={'fontSize': '0.85rem', 'marginTop': '1rem', 'padding': '0.5rem', 
                             'backgroundColor': 'rgba(94, 234, 212, 0.1)', 'borderRadius': '8px'})
                ])
            ])
        ], className="analysis-card",)
        
        analysis_cards.append(index_card)
        
        # 6. 한국 특화 환율 인덱스 설명 카드 (전체 너비)
        index_explanation_card = html.Div([
            html.H3("📖 KRW 인덱스 해석 가이드"),
            
            html.Div([
                html.H4("인덱스 이해하기", style={'color': '#5eead4', 'fontSize': '1rem', 'marginBottom': '0.5rem'}),
                html.Div([
                    html.P("KRW 인덱스는 원화 가치에 영향을 미치는 6가지 핵심 요소를 종합한 지표입니다.", 
                          style={'fontSize': '0.85rem', 'marginBottom': '10px'}),
                    
                    html.H5("📈 지수 읽는 법", style={'color': '#8b5cf6', 'fontSize': '0.9rem', 'marginTop': '10px'}),
                    html.Ul([
                        html.Li("100 = 평균 수준", style={'fontSize': '0.8rem'}),
                        html.Li("110 이상 = 원화 약세 압력", style={'fontSize': '0.8rem'}),
                        html.Li("90 이하 = 원화 강세 압력", style={'fontSize': '0.8rem'}),
                    ], style={'marginLeft': '10px', 'marginBottom': '10px'})
                ], style={'padding': '8px', 'backgroundColor': 'rgba(94, 234, 212, 0.05)', 
                         'borderRadius': '6px', 'marginBottom': '10px'}),
                
                html.H4("주요 영향 요인", style={'color': '#fbbf24', 'fontSize': '1rem', 'marginTop': '1rem', 'marginBottom': '0.5rem'}),
                html.Div([
                    html.Table([
                        html.Tbody([
                            html.Tr([
                                html.Td("🇺🇸", style={'fontSize': '1.2rem', 'padding': '5px'}),
                                html.Td("DXY ↑", style={'fontSize': '0.8rem', 'padding': '5px', 'fontWeight': 'bold'}),
                                html.Td("→ 달러 강세 → 원화 약세", style={'fontSize': '0.75rem', 'padding': '5px', 'color': '#f87171'})
                            ]),
                            html.Tr([
                                html.Td("📈", style={'fontSize': '1.2rem', 'padding': '5px'}),
                                html.Td("KOSPI ↑", style={'fontSize': '0.8rem', 'padding': '5px', 'fontWeight': 'bold'}),
                                html.Td("→ 외국인 투자 ↑ → 원화 강세", style={'fontSize': '0.75rem', 'padding': '5px', 'color': '#5eead4'})
                            ]),
                            html.Tr([
                                html.Td("😨", style={'fontSize': '1.2rem', 'padding': '5px'}),
                                html.Td("VIX ↑", style={'fontSize': '0.8rem', 'padding': '5px', 'fontWeight': 'bold'}),
                                html.Td("→ 리스크 회피 → 원화 약세", style={'fontSize': '0.75rem', 'padding': '5px', 'color': '#f87171'})
                            ]),
                            html.Tr([
                                html.Td("🛢️", style={'fontSize': '1.2rem', 'padding': '5px'}),
                                html.Td("원유 ↑", style={'fontSize': '0.8rem', 'padding': '5px', 'fontWeight': 'bold'}),
                                html.Td("→ 수입 부담 ↑ → 원화 약세", style={'fontSize': '0.75rem', 'padding': '5px', 'color': '#f87171'})
                            ]),
                            html.Tr([
                                html.Td("🏆", style={'fontSize': '1.2rem', 'padding': '5px'}),
                                html.Td("금 ↑", style={'fontSize': '0.8rem', 'padding': '5px', 'fontWeight': 'bold'}),
                                html.Td("→ 달러 약세 → 원화 강세", style={'fontSize': '0.75rem', 'padding': '5px', 'color': '#5eead4'})
                            ]),
                            html.Tr([
                                html.Td("💰", style={'fontSize': '1.2rem', 'padding': '5px'}),
                                html.Td("美금리 ↑", style={'fontSize': '0.8rem', 'padding': '5px', 'fontWeight': 'bold'}),
                                html.Td("→ 금리차 확대 → 원화 약세", style={'fontSize': '0.75rem', 'padding': '5px', 'color': '#f87171'})
                            ])
                        ])
                    ], style={'width': '100%'})
                ], style={'padding': '8px', 'backgroundColor': 'rgba(139, 92, 246, 0.05)', 
                         'borderRadius': '6px'}),
                
                html.H4("투자 전략 활용", style={'color': '#f87171', 'fontSize': '1rem', 'marginTop': '1rem', 'marginBottom': '0.5rem'}),
                html.Div([
                    html.Div([
                        html.H5("🔴 +2σ 이상 (강한 과매수)", style={'fontSize': '0.85rem', 'color': '#ef4444'}),
                        html.P("• 원화 매우 약세", style={'fontSize': '0.75rem', 'marginLeft': '15px'}),
                        html.P("• 반전 임박 가능", style={'fontSize': '0.75rem', 'marginLeft': '15px'}),
                        html.P("• 달러 매도 타이밍", style={'fontSize': '0.75rem', 'marginLeft': '15px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '10px', 'padding': '5px', 'borderLeft': '3px solid #ef4444'}),
                    
                    html.Div([
                        html.H5("🟡 +1σ ~ +2σ (과매수)", style={'fontSize': '0.85rem', 'color': '#fbbf24'}),
                        html.P("• 조정 가능성 관찰", style={'fontSize': '0.75rem', 'marginLeft': '15px'}),
                        html.P("• 단계적 환전 고려", style={'fontSize': '0.75rem', 'marginLeft': '15px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '10px', 'padding': '5px', 'borderLeft': '3px solid #fbbf24'}),
                    
                    html.Div([
                        html.H5("⚪ -1σ ~ +1σ (중립)", style={'fontSize': '0.85rem', 'color': '#9ca3af'}),
                        html.P("• 정상 범위", style={'fontSize': '0.75rem', 'marginLeft': '15px'}),
                        html.P("• 추세 관망", style={'fontSize': '0.75rem', 'marginLeft': '15px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '10px', 'padding': '5px', 'borderLeft': '3px solid #9ca3af'}),
                    
                    html.Div([
                        html.H5("🟢 -2σ 이하 (강한 과매도)", style={'fontSize': '0.85rem', 'color': '#5eead4'}),
                        html.P("• 원화 매우 강세", style={'fontSize': '0.75rem', 'marginLeft': '15px'}),
                        html.P("• 반등 임박 가능", style={'fontSize': '0.75rem', 'marginLeft': '15px'}),
                        html.P("• 달러 매수 타이밍", style={'fontSize': '0.75rem', 'marginLeft': '15px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '5px', 'padding': '5px', 'borderLeft': '3px solid #5eead4'})
                ], style={'fontSize': '0.8rem'}),
                
                html.Div([
                    html.P("⚠️ 주의: 지표는 참고용이며 실제 투자는 종합적 판단 필요", 
                          style={'fontSize': '0.75rem', 'color': '#ef4444', 'fontWeight': 'bold', 'textAlign': 'center'})
                ], style={'marginTop': '1rem', 'padding': '8px', 'backgroundColor': 'rgba(239, 68, 68, 0.1)', 
                         'borderRadius': '6px', 'border': '1px solid rgba(239, 68, 68, 0.3)'})
            ])
        ], className="analysis-card", )
        
        analysis_cards.append(index_explanation_card)
        
        # 데이터 저장 (날짜를 문자열로 변환)
        data_dict = {
            'data': data.reset_index().to_dict(orient='records'),
            'timestamp': datetime.now().isoformat()
        }
        
        models_dict = {
            'predictions': {
                k: v.reset_index().to_dict(orient='records')
                for k, v in models.predictions.items()
            },
            'performance': models.performance,
            'future_predictions': future_predictions
        }
        
        # detector_dict의 날짜를 문자열로 변환
        alerts_serializable = []
        for alert in detector.alerts:
            alert_copy = alert.copy()
            if 'date' in alert_copy and hasattr(alert_copy['date'], 'isoformat'):
                alert_copy['date'] = alert_copy['date'].isoformat()
            alerts_serializable.append(alert_copy)
        
        bubble_periods_serializable = []
        for period in bubble_periods:
            period_copy = period.copy()
            if 'start' in period_copy and hasattr(period_copy['start'], 'isoformat'):
                period_copy['start'] = period_copy['start'].isoformat()
            if 'end' in period_copy and hasattr(period_copy['end'], 'isoformat'):
                period_copy['end'] = period_copy['end'].isoformat()
            bubble_periods_serializable.append(period_copy)
        
        detector_dict = {
            'alerts': alerts_serializable,
            'bubble_periods': bubble_periods_serializable,
            'risk_level': risk_level,
            'risk_score': float(risk_score),  # numpy float을 일반 float으로 변환
            'dtw_anomalies': [
                {**a, 'date': a['date'].isoformat() if hasattr(a.get('date'), 'isoformat') else a.get('date')} 
                for a in (detector.dtw_anomalies if hasattr(detector, 'dtw_anomalies') else [])
            ],
            'dtw_timeline': {
                str(k): float(v) for k, v in detector.dtw_timeline.to_dict().items()
            } if hasattr(detector, 'dtw_timeline') and detector.dtw_timeline is not None and len(detector.dtw_timeline) > 0 else {}
        }
        
        status_msg = f"✅ 실제 데이터 분석 완료 (DTW+IF) ({datetime.now().strftime('%H:%M:%S')})"
        
        return alert_banner, forecast_section, summary_cards, analysis_cards, status_msg, data_dict, models_dict, detector_dict
        
    except Exception as e:
        error_msg = html.Div([
            html.H4("❌ 오류 발생"),
            html.P(str(e)),
            html.P("인터넷 연결과 yfinance 패키지를 확인하세요.")
        ], className="error-message")
        return [], [], [error_msg], [], f"❌ 오류: {str(e)}", {}, {}, {}


# ============================================
# 메인 실행
# ============================================
if __name__ == '__main__':
    import socket
    
    def find_available_port(start_port=8050, max_attempts=20):
        for port in range(start_port, start_port + max_attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError("사용 가능한 포트를 찾을 수 없습니다")
    
    port = find_available_port()
    print(f"{'='*60}")
    print(f"🚀 실제 KRW/USD 환율 예측 대시보드 (DTW + IF)")
    print(f"{'='*60}")
    print(f"📊 실제 Yahoo Finance 데이터를 사용합니다")
    print(f"🎯 Isolation Forest + Dynamic Time Warping 이상치 탐지")
    print(f"🌐 브라우저에서 접속: http://localhost:{port}")
    print(f"⏹️  종료하려면 Ctrl+C를 누르세요")
    print(f"{'='*60}")
    
    app.run(debug=False, host='0.0.0.0', port=port)