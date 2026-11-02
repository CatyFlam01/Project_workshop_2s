"""
ML-based Habit Recommendation Engine
=====================================
Принимает результаты онбординг-теста в JSON, формирует вектор предпочтений
и рекомендует 3 привычки из разных сфер с использованием:
- Векторного представления с нормализацией (L2)
- Cosine Similarity для поиска похожих профилей
- Жадного алгоритма диверсификации (разные категории)
- Опционально: KNN для расширения пула кандидатов
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from numpy.linalg import norm

# ============================================================================
# КОНФИГУРАЦИЯ: КАТЕГОРИИ И ВЕСА ИЗ ТЕСТА
# ============================================================================

# Категории, которые используются в тесте онбординга
ONBOARDING_CATEGORIES = [
    "health",
    "productivity",
    # 'sleep',
    "self_development",
    "learning",
    "mindfulness",
    "sport",
]

# Маппинг категорий теста на категории из БД (для совместимости с
# tags/init.sql)
CATEGORY_MAPPING = {
    "health": "здоровье",
    "productivity": "продуктивность",
    "sleep": "здоровье",
    "mindfulness": "ментальное состояние",
    "learning": "обучение",
}

# База привычек с их векторами (симуляция "БД" для рекомендаций)
# В продакшене это можно загружать из Django ORM: Habit.objects.all()
HABIT_DATABASE: List[Dict] = [
    # Физические привычки (здоровье/фитнес)
    {
        "id": "run_morning",
        "title": "Утренняя пробежка 10 мин",
        "icon": "dumbbell",  # Спорт
        "tags": ["бег", "зарядка"],
        "category": "здоровье",
        "time_required": "10 мин",
        "best_time": ["morning"],
        "vector": {"fitness": 1.0, "health": 0.9, "productivity": 0.3},
    },
    {
        "id": "stretching",
        "title": "Растяжка после пробуждения",
        "icon": "dumbbell",  # Спорт
        "tags": ["растяжка", "зарядка"],
        "category": "здоровье",
        "time_required": "5-10 мин",
        "best_time": ["morning"],
        "vector": {"fitness": 0.7, "health": 0.8, "mindfulness": 0.4},
    },
    {
        "id": "water_intake",
        "title": "Выпивать стакан воды утром",
        "icon": "water",  # Вода
        "tags": ["вода"],
        "category": "здоровье",
        "time_required": "1 мин",
        "best_time": ["morning"],
        "vector": {"health": 1.0, "productivity": 0.2},
    },
    # Обучение и развитие
    {
        "id": "read_book",
        "title": "Читать 10 страниц книги",
        "icon": "book",  # Чтение
        "tags": ["чтение (книги)"],
        "category": "обучение",
        "time_required": "15 мин",
        "best_time": ["evening", "night"],
        "vector": {
            "learning": 1.0,
            "self_development": 0.8,
            "mindfulness": 0.3,
        },
    },
    {
        "id": "language_practice",
        "title": "Практиковать иностранный язык",
        "icon": "brain",  # Мышление
        "tags": ["изучение языка"],
        "category": "обучение",
        "time_required": "15 мин",
        "best_time": ["morning", "evening"],
        "vector": {
            "learning": 1.0,
            "self_development": 0.9,
            "productivity": 0.4,
        },
    },
    {
        "id": "online_course",
        "title": "Пройти урок онлайн-курса",
        "icon": "brain",  # Мышление
        "tags": ["онлайн-курс"],
        "category": "обучение",
        "time_required": "30 мин",
        "best_time": ["evening"],
        "vector": {
            "learning": 1.0,
            "productivity": 0.5,
            "self_development": 0.7,
        },
    },
    # Продуктивность
    {
        "id": "daily_planning",
        "title": "Составлять план на день",
        "icon": "brain",  # Мышление
        "tags": ["планирование"],
        "category": "продуктивность",
        "time_required": "5 мин",
        "best_time": ["morning"],
        "vector": {
            "productivity": 1.0,
            "self_development": 0.6,
            "mindfulness": 0.3,
        },
    },
    {
        "id": "deep_work_session",
        "title": "25 минут фокус работы (Pomodoro)",
        "icon": "code",  # Работа
        "tags": ["deep work"],
        "category": "продуктивность",
        "time_required": "25 мин",
        "best_time": ["morning", "afternoon"],
        "vector": {
            "productivity": 1.0,
            "learning": 0.4,
            "self_development": 0.5,
        },
    },
    {
        "id": "budget_tracking",
        "title": "Записывать расходы дня",
        "icon": "brain",  # Мышление
        "tags": ["бюджет / учёт денег"],
        "category": "продуктивность",
        "time_required": "5 мин",
        "best_time": ["evening"],
        "vector": {
            "finance": 1.0,
            "productivity": 0.8,
            "self_development": 0.4,
        },
    },
    # Ментальное состояние / осознанность
    {
        "id": "meditation_5min",
        "title": "Медитация 5 минут",
        "icon": "spa",  # Йога (подходит для медитации)
        "tags": ["медитация", "дыхательная практика"],
        "category": "ментальное состояние",
        "time_required": "5 мин",
        "best_time": ["morning", "evening"],
        "vector": {"mindfulness": 1.0, "sleep": 0.6, "health": 0.4},
    },
    {
        "id": "gratitude_journal",
        "title": "Записать 3 вещи, за которые благодарен",
        "icon": "heart",  # Здоровье (душевное)
        "tags": ["благодарность", "дневник"],
        "category": "ментальное состояние",
        "time_required": "5 мин",
        "best_time": ["evening"],
        "vector": {"mindfulness": 1.0, "self_development": 0.7, "sleep": 0.5},
    },
    {
        "id": "screen_free_hour",
        "title": "Час без экрана перед сном",
        "icon": "leaf",  # Природа (отдых, спокойствие)
        "tags": ["отдых без экрана", "цифровой детокс"],
        "category": "ментальное состояние",
        "time_required": "60 мин",
        "best_time": ["evening", "night"],
        "vector": {"mindfulness": 0.9, "sleep": 1.0, "health": 0.5},
    },
    # Сон
    {
        "id": "sleep_schedule",
        "title": "Ложиться спать до 23:00",
        "icon": "moon",  # Сон
        "tags": ["ранний сон"],
        "category": "здоровье",
        "time_required": "0 мин",
        "best_time": ["night"],
        "vector": {"sleep": 1.0, "health": 0.9, "productivity": 0.6},
    },
    # Социальное / личная жизнь
    {
        "id": "call_friend",
        "title": "Позвонить другу или родным",
        "icon": "heart",  # Здоровье (социальные связи)
        "tags": ["общение (звонок)", "друзья (вживую)"],
        "category": "личная жизнь",
        "time_required": "15 мин",
        "best_time": ["evening"],
        "vector": {"mindfulness": 0.6, "self_development": 0.4, "health": 0.3},
    },
]


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ВЕКТОРОВ
# ============================================================================


def normalize_vector(vec: Dict[str, float]) -> np.ndarray:
    """
    Преобразует словарь {category: score} в нормализованный numpy-вектор.
    Использует L2-нормализацию для корректного сравнения через cosine
    similarity.
    """
    vector = np.array([vec.get(cat, 0.0) for cat in ONBOARDING_CATEGORIES])
    norm_val = norm(vector)
    if norm_val == 0:
        return vector  # нулевой вектор остаётся нулевым
    return vector / norm_val


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Вычисляет косинусное сходство между двумя векторами.
    Возвращает значение от -1 до 1 (чем ближе к 1 — тем более похожи).
    """
    norm1, norm2 = norm(vec1), norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def build_user_vector(test_results: Dict) -> np.ndarray:
    """
    Формирует вектор предпочтений пользователя из результатов теста.

    Args:
        test_results: JSON с ответами на вопросы онбординга
            {
                "goals": ["productivity", "health"],  # multi-select
                "sport_frequency": "1-2 times/week",    # single-select
                "difficulties": ["concentration"],      # multi-select
                "free_time": "15 minutes",              # single-select
                "preferred_time": ["evening"],          # multi-select
            }

    Returns:
        Нормализованный numpy-вектор размерности 8 (по количеству категорий)
    """
    # Инициализируем накопитель весов
    scores = {cat: 0.0 for cat in ONBOARDING_CATEGORIES}

    # 1. Интересы (Question 1) — multi-select, самый сильный сигнал
    interest_weights = {
        "sport": ("sport", 1.0),
        "learning": ("learning", 1.0),
        "health": ("health", 1.0),
        "mindfulness": ("mindfulness", 1.0),
        "productivity": ("productivity", 1.0),
        "self_development": ("self_development", 1.0),
        # 'reading': ('learning', 1.0),
        # 'meditation': ('mindfulness', 1.0),
        # 'sleep': ('sleep', 1.0),
        # 'nutrition': ('health', 1.0), 'languages': ('learning', 1.0),
        # 'productivity': ('productivity', 1.0),
        # 'finance': ('finance', 1.0), 'self_development':
        # ('self_development', 1.0)
    }
    for interest in test_results.get("goals", []):
        if interest in interest_weights:
            cat, weight = interest_weights[interest]
            scores[cat] += weight * 1.5  # усиленный вес для явных интересов

    # 2. Частота спорта (Question 2) — single-select
    sport_weights = {
        "never": 0.2,
        "1-2 times/week": 0.4,
        "3-5 times/week": 0.7,
        "every day": 1.0,
    }
    sport_freq = test_results.get("sport_frequency", "never")
    if isinstance(sport_freq, list):
        sport_freq = sport_freq[0] if sport_freq else "never"
    scores["sport"] += sport_weights.get(sport_freq, 0.2)

    # 3. Сложности (Question 3) — multi-select
    difficulty_weights = {
        "wake_up": ("health", 0.8),
        "concentration": ("productivity", 0.8),
        "exercise": ("sport", 0.8),
        "drink_water": ("health", 0.8),
        "read": ("learning", 0.8),
        "rest": ("mindfulness", 0.8),
    }
    for diff in test_results.get("difficulties", []):
        if diff in difficulty_weights:
            cat, weight = difficulty_weights[diff]
            scores[cat] += weight

    # Нормализуем итоговый вектор
    return normalize_vector(scores)


# ============================================================================
# ЯДРО РЕКОМЕНДАТЕЛЬНОЙ СИСТЕМЫ
# ============================================================================


@dataclass
class RecommendationConfig:
    """Конфигурация для тонкой настройки рекомендаций."""

    n_recommendations: int = 3
    require_diversity: bool = True  # выбирать из разных категорий
    min_similarity_threshold: float = 0.1  # порог отсечения слабых совпадений
    time_filter_enabled: bool = True  # учитывать preferred_time пользователя
    respect_user_interests: bool = True  # приоритет указанным интересам


def _filter_by_time(
    habits: List[Dict], preferred_times: List[str]
) -> List[Dict]:
    """Фильтрует привычки по предпочтительному времени суток."""
    if not preferred_times:
        return habits

    time_mapping = {
        "morning": "morning",
        "утро": "morning",
        "afternoon": "afternoon",
        "день": "afternoon",
        "evening": "evening",
        "вечер": "evening",
        "night": "night",
        "ночь": "night",
    }

    normalized_times = [
        time_mapping.get(t.lower(), t.lower()) for t in preferred_times
    ]

    filtered = []
    for habit in habits:
        habit_times = habit.get("best_time", [])
        # Если есть пересечение — оставляем привычку
        if any(t in normalized_times for t in habit_times) or not habit_times:
            filtered.append(habit)
    return filtered


def _select_diverse_habits(
    scored_habits: List[tuple],  # [(habit, similarity_score), ...]
    n: int,
    categories: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Жадный алгоритм выбора разнообразных привычек.
    Старается взять по одной из разных категорий при высокой схожести.
    """
    if not scored_habits:
        return []

    selected = []
    used_categories = set()

    # Сортируем по убыванию сходства
    scored_habits_sorted = sorted(
        scored_habits, key=lambda x: x[1], reverse=True
    )

    for habit, _score in scored_habits_sorted:
        if len(selected) >= n:
            break

        category = habit.get("category", "unknown")

        # Если включена диверсификация — пропускаем уже выбранную категорию
        if categories and category in used_categories:
            continue

        selected.append(habit)
        used_categories.add(category)

    # Если не набрали enough — добавляем оставшиеся без проверки категорий
    if len(selected) < n:
        for habit, _score in scored_habits_sorted:
            if habit in selected:
                continue
            selected.append(habit)
            if len(selected) >= n:
                break

    return selected[:n]


def recommend_habits(
    test_results: Union[str, Dict],
    config: Optional[RecommendationConfig] = None,
    custom_habit_db: Optional[List[Dict]] = None,
) -> Dict:
    """
    🔑 ОСНОВНАЯ ФУНКЦИЯ — вызывается одним вызовом

    Принимает результаты теста и возвращает 3 персонализированные привычки.

    Args:
        test_results: JSON-строка или dict с ответами пользователя
        config: Опциональная конфигурация (см. RecommendationConfig)
        custom_habit_db: Опциональная кастомная БД привычек (для тестов)

    Returns:
        Dict с рекомендациями и метаданными:
        {
            "recommendations": [habit1, habit2, habit3],
            "user_vector": {...},  # нормализованные веса категорий
            "similarity_scores": {...},  # для отладки
            "metadata": {"diversity_applied": bool, "time_filtered": bool}
        }

    Пример входных данных:
    {
        "goals": ["productivity", "health"],
        "sport_frequency": "1-2 times/week",
        "difficulties": ["concentration"],
        "free_time": "15 minutes",
        "preferred_time": ["evening"],
    }
    """
    # 0. Подготовка
    cfg = config or RecommendationConfig()
    if cfg.n_recommendations < 1:
        raise ValueError("n_recommendations must be positive")

    # Парсим JSON если пришла строка
    if isinstance(test_results, str):
        test_results = json.loads(test_results)

    habit_db = (
        custom_habit_db if custom_habit_db is not None else HABIT_DATABASE
    )

    # 1. Формируем вектор пользователя
    user_vector = build_user_vector(test_results)
    user_vector_dict = dict(
        zip(ONBOARDING_CATEGORIES, user_vector.tolist(), strict=False)
    )

    # 2. Фильтруем по времени (если включено)
    preferred_times = test_results.get("preferred_time", [])
    candidates = (
        _filter_by_time(habit_db, preferred_times)
        if cfg.time_filter_enabled
        else habit_db
    )

    # 3. Вычисляем similarity для каждой привычки
    scored = []
    for habit in candidates:
        habit_vector_raw = habit.get("vector", {})
        # Приводим вектор привычки к той же схеме категорий
        habit_vector = normalize_vector(
            {
                cat: habit_vector_raw.get(cat, 0.0)
                for cat in ONBOARDING_CATEGORIES
            }
        )

        sim = cosine_similarity(user_vector, habit_vector)

        # Бустим привычки, которые явно в интересах пользователя
        if cfg.respect_user_interests:
            user_interests = test_results.get("goals", [])
            habit_tags = habit.get("tags", [])
            # Простая эвристика: если тег привычки совпадает с интересом
            interest_boost = (
                0.15
                if any(
                    tag.lower() in [i.lower() for i in user_interests]
                    for tag in habit_tags
                )
                else 0.0
            )
            sim += interest_boost

        if sim >= cfg.min_similarity_threshold:
            scored.append((habit, sim))

    # 4. Отбираем разнообразные рекомендации
    recommendations = _select_diverse_habits(
        scored,
        n=cfg.n_recommendations,
        categories=ONBOARDING_CATEGORIES if cfg.require_diversity else None,
    )

    # 5. Формируем ответ
    return {
        "recommendations": recommendations,
        "user_vector": {
            cat: round(float(val), 3)
            for cat, val in user_vector_dict.items()
            if val > 0.01
        },
        "similarity_scores": {
            h["id"]: round(s, 3)
            for h, s in sorted(scored, key=lambda item: item[1], reverse=True)[
                :10
            ]  # топ-10 для отладки
        },
        "metadata": {
            "diversity_applied": cfg.require_diversity,
            "time_filtered": cfg.time_filter_enabled and bool(preferred_times),
            "total_candidates": len(candidates),
            "scored_above_threshold": len(scored),
        },
    }


# ============================================================================
# ДОПОЛНИТЕЛЬНЫЕ МЕТОДЫ (по запросу можно расширить)
# ============================================================================


def find_similar_user_profiles(
    user_vector: np.ndarray, profile_database: List[Dict], k: int = 5
) -> List[Dict]:
    """
    KNN-поиск: находит k наиболее похожих профилей пользователей.
    Можно использовать для collaborative filtering в будущем.

    Args:
        user_vector: вектор текущего пользователя
    profile_database: список { "user_id": ..., "vector": {...},
    "liked_habits": [...] }
        k: количество ближайших соседей

    Returns:
        Список профилей с добавленным полем 'similarity'
    """
    scored_profiles = []
    for profile in profile_database:
        profile_vec = normalize_vector(profile.get("vector", {}))
        sim = cosine_similarity(user_vector, profile_vec)
        scored_profiles.append({**profile, "similarity": sim})

    scored_profiles.sort(key=lambda x: x["similarity"], reverse=True)
    return scored_profiles[:k]


def generate_synthetic_profiles(n: int = 100) -> List[Dict]:
    """
    Генерирует синтетические профили для тестирования / cold start.
    Каждый профиль — случайная комбинация предпочтений по категориям.
    """
    profiles = []
    for i in range(n):
        # Случайные веса с экспоненциальным распределением (имитация
        # реальных данных)
        weights = np.random.exponential(
            scale=0.5, size=len(ONBOARDING_CATEGORIES)
        )
        weights = weights / (norm(weights) + 1e-8)  # нормализация

        profile_vector = dict(
            zip(ONBOARDING_CATEGORIES, weights.tolist(), strict=False)
        )

        # Случайные "понравившиеся" привычки для collaborative filtering
        n_liked = np.random.randint(2, 6)
        liked = np.random.choice(
            [h["id"] for h in HABIT_DATABASE],
            size=min(n_liked, len(HABIT_DATABASE)),
            replace=False,
        ).tolist()

        profiles.append(
            {
                "user_id": f"synthetic_{i}",
                "vector": profile_vector,
                "liked_habits": liked,
            }
        )

    return profiles


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ (для тестов / демо)
# ============================================================================

if __name__ == "__main__":
    # Пример входных данных из вашего описания
    sample_test = {
        "goals": ["productivity"],
        "sport_frequency": "1-2 times/week",
        "difficulties": ["concentration"],
        "free_time": "15 minutes",
        "preferred_time": ["evening"],
    }

    result = recommend_habits(sample_test)

    print("🎯 Рекомендации для пользователя:\n")
    for i, habit in enumerate(result["recommendations"], 1):
        print(f"{i}. {habit['title']}")
        print(f"   🏷  Категория: {habit['category']}")
        print(f"   ⏱  Время: {habit['time_required']}")
        print(f"   🕐 Лучше всего: {', '.join(habit['best_time'])}")
        print()

    print("📊 Вектор предпочтений (нормализованный):")
    for cat, val in result["user_vector"].items():
        print(f"   {cat}: {val:.3f}")
