# EduShorts

EduShorts is a personalized micro-learning platform designed to help students learn faster through short-form educational content and adaptive recommendations.

The project combines a mobile-first learning experience with a hybrid recommendation system that selects the most relevant algorithm based on user context (cold-start users, active users, and similarity-based exploration).

## Overview

EduShorts focuses on three core outcomes:

- **Efficient learning** through short videos and audio-first content
- **Personalized discovery** powered by multiple recommendation strategies
- **Consistent engagement** through gamified progression loops

## Core Features

### Learning Experience
- Short vertical lesson videos with swipe-based navigation
- Podcast-style audio lessons for passive learning moments
- Mobile-first interface optimized for fast, low-friction study sessions

### Engagement Layer
- Streak tracking for daily learning consistency
- Badge and level progression tied to watch/listen activity
- Motivational feedback loops inspired by modern learning products

### Recommendation System (Primary Project Focus)
EduShorts uses a **hybrid recommendation architecture** that dynamically routes users to different recommenders depending on available behavioral and content signals.

#### 1) FAISS-Based Similarity Retrieval
- Fast approximate nearest-neighbor search over content embeddings
- Supports “more like this” exploration flows
- Low-latency candidate generation for related lessons

#### 2) ALS Collaborative Filtering
- Learns latent user-item preferences from interaction patterns
- Performs best for users with sufficient engagement history
- Improves ranking quality for active users with established interests

#### 3) Content-Based Filtering
- Uses metadata/feature similarity to match content with user interests
- Effective for cold-start users and new content discovery
- Provides stable personalization when collaborative signals are sparse

#### 4) Hybrid Orchestration Strategy
- Selects recommendation path based on user state and data availability
- Combines candidate sets and ranking signals where appropriate
- Designed for practical production evolution rather than single-model dependency

## Recommendation Logic by User Scenario

- **New user (cold start):** Content-Based first, then FAISS similarity expansion
- **Active user:** ALS-driven ranking with similarity-based backfilling
- **Topic explorer:** FAISS-heavy retrieval around recent interactions

This scenario-aware design is the key technical differentiator of the project.

## Technical Stack

- **Frontend:** React Native (Expo), TypeScript
- **Recommendation Layer:** FAISS, ALS, Content-Based Filtering (Hybrid)
- **Backend:** In progress (current build demonstrates functional frontend + recommendation integration prototype)

## Project Structure (High Level)

- Mobile client for content feed, playback, and engagement interactions
- Recommendation module for candidate retrieval and ranking strategies
- Integration layer for routing logic between recommender components

## Current Status

- Functional mobile prototype available
- Recommendation integration implemented at prototype level
- Backend services and production-grade data pipelines are planned next steps

## Local Setup

```bash
# Clone repository
git clone https://github.com/b2230765034/EduShorts.git
cd EduShorts

# Install dependencies
npm install

# Start Expo app
npx expo start
```

## Demo

Project demo link: *(to be added)*

## Why This Project Matters

EduShorts demonstrates end-to-end product thinking:
- user-centric mobile UX,
- practical recommender-system design,
- and engagement mechanics that support sustained learning behavior.

For portfolio purposes, the recommendation-system architecture is the central engineering contribution of this project.
