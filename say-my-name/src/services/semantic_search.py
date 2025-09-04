"""
AI Assistant Semantic Search Engine

This module provides advanced semantic search capabilities with sophisticated relevance scoring,
query analysis, and context filtering. It enhances the basic similarity search with intelligent
ranking, temporal awareness, and multi-dimensional relevance assessment.

Key Concepts:
- Multi-Factor Relevance: Combines semantic similarity, recency, and context quality
- Query Analysis: Intelligent preprocessing of search queries for better results
- Temporal Scoring: Recent conversations weighted higher for relevance
- Context Quality: Assesses the richness and usefulness of retrieved memories
- Search Result Ranking: Advanced algorithms for ordering results by overall utility

Theory Background:
Basic vector similarity (cosine distance) provides semantic matching, but real-world
relevance requires multiple factors:

1. Semantic Similarity: How closely related is the content?
2. Temporal Relevance: How recent and contextually fresh is the information?
3. Content Quality: How detailed and useful is the retrieved content?
4. Conversation Flow: How well does it fit the current discussion context?

This engine combines these factors using weighted scoring to provide the most
useful memories for enhancing AI responses.

Usage Example:
    from services.semantic_search import SemanticSearchEngine, init_search_engine
    
    # Initialize search engine
    search_engine = init_search_engine()
    
    # Perform enhanced search
    results = search_engine.search_memories(
        conversation_id="conv_123",
        query="FastAPI debugging",
        max_results=3,
        include_context_analysis=True
    )
    
    for result in results:
        print(f"Score: {result.relevance_score:.3f}")
        print(f"Memory: {result.memory.get_context_summary()}")

Architecture:
SearchConfig - Configuration for search behavior and scoring weights
RelevanceScorer - Multi-factor scoring algorithm for search results  
QueryAnalyzer - Intelligent query preprocessing and enhancement
SearchResult - Enhanced result with relevance metadata
SemanticSearchEngine - Main search interface with advanced capabilities
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field, ConfigDict

# Import our services
try:
    from .memory_service import MemoryService, MessageMemory
    from .embedding_service import EmbeddingService
except ImportError:
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from services.memory_service import MemoryService, MessageMemory
        from services.embedding_service import EmbeddingService
    except ImportError:
        from src.services.memory_service import MemoryService, MessageMemory
        from src.services.embedding_service import EmbeddingService


class SearchConfig(BaseModel):
    """
    Configuration for semantic search engine behavior.
    
    Defines scoring weights, filtering parameters, and search optimization
    settings for the enhanced semantic search functionality.
    
    Attributes:
        similarity_weight: Weight for semantic similarity score (0.0-1.0)
        recency_weight: Weight for temporal recency score (0.0-1.0)
        quality_weight: Weight for content quality score (0.0-1.0)
        context_weight: Weight for conversation context fit (0.0-1.0)
        recency_decay_hours: Hours for recency score to decay to 50%
        min_content_length: Minimum character length for quality scoring
        max_search_results: Maximum results to return from search
        enable_query_expansion: Whether to expand queries with related terms
        boost_recent_threshold_hours: Hours to consider "recent" for boosting
    """
    
    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True
    )
    
    similarity_weight: float = Field(
        default=0.4,
        description="Weight for semantic similarity in final score",
        ge=0.0,
        le=1.0
    )
    
    recency_weight: float = Field(
        default=0.3,
        description="Weight for temporal recency in final score",
        ge=0.0,
        le=1.0
    )
    
    quality_weight: float = Field(
        default=0.2,
        description="Weight for content quality in final score",
        ge=0.0,
        le=1.0
    )
    
    context_weight: float = Field(
        default=0.1,
        description="Weight for conversation context fit in final score",
        ge=0.0,
        le=1.0
    )
    
    recency_decay_hours: float = Field(
        default=168.0,  # 1 week
        description="Hours for recency score to decay to 50%",
        gt=0.0
    )
    
    min_content_length: int = Field(
        default=20,
        description="Minimum content length for quality consideration",
        gt=0
    )
    
    max_search_results: int = Field(
        default=10,
        description="Maximum number of search results to return",
        gt=0,
        le=50
    )
    
    enable_query_expansion: bool = Field(
        default=True,
        description="Enable intelligent query expansion with related terms"
    )
    
    boost_recent_threshold_hours: float = Field(
        default=24.0,
        description="Hours to consider content 'recent' for score boosting",
        gt=0.0
    )
    
    def validate_weights(self) -> bool:
        """
        Validate that all weights sum to approximately 1.0.
        
        Returns:
            bool: True if weights are properly balanced
        """
        total_weight = (
            self.similarity_weight + 
            self.recency_weight + 
            self.quality_weight + 
            self.context_weight
        )
        return abs(total_weight - 1.0) < 0.01  # Allow small floating point variance


class RelevanceScorer:
    """
    Multi-factor scoring algorithm for semantic search results.
    
    Combines semantic similarity with temporal relevance, content quality,
    and contextual fit to produce comprehensive relevance scores.
    """
    
    def __init__(self, config: SearchConfig):
        """
        Initialize relevance scorer with configuration.
        
        Args:
            config: SearchConfig with scoring weights and parameters
        """
        self.config = config
        
        if not config.validate_weights():
            print("Warning: Search scoring weights don't sum to 1.0")
    
    def calculate_similarity_score(self, similarity: float) -> float:
        """
        Calculate normalized similarity score.
        
        Args:
            similarity: Raw cosine similarity (0.0-1.0)
            
        Returns:
            float: Normalized similarity score (0.0-1.0)
        """
        # Similarity is already normalized (0.0-1.0)
        return max(0.0, min(1.0, similarity))
    
    def calculate_recency_score(self, memory_timestamp: datetime) -> float:
        """
        Calculate temporal recency score with exponential decay.
        
        Recent memories are scored higher, with score decaying exponentially
        over time according to the configured decay parameter.
        
        Args:
            memory_timestamp: When the memory was created
            
        Returns:
            float: Recency score (0.0-1.0)
        """
        now = datetime.now()
        hours_ago = (now - memory_timestamp).total_seconds() / 3600
        
        if hours_ago < 0:
            return 1.0  # Future timestamps get max score
        
        # Exponential decay: score = e^(-t/half_life)
        # At decay_hours, score = 0.5
        decay_constant = 0.693 / self.config.recency_decay_hours  # ln(2) / half_life
        recency_score = np.exp(-decay_constant * hours_ago)
        
        # Boost very recent content
        if hours_ago <= self.config.boost_recent_threshold_hours:
            boost_factor = 1.2  # 20% boost for recent content
            recency_score = min(1.0, recency_score * boost_factor)
        
        return recency_score
    
    def calculate_quality_score(self, memory: MessageMemory) -> float:
        """
        Calculate content quality score based on multiple factors.
        
        Assesses the richness and usefulness of the memory content
        including length, detail, and information density.
        
        Args:
            memory: MessageMemory to assess for quality
            
        Returns:
            float: Quality score (0.0-1.0)
        """
        quality_factors = []
        
        # Factor 1: Content length (longer generally better, up to a point)
        combined_length = len(memory.user_message) + len(memory.ai_response)
        if combined_length < self.config.min_content_length:
            length_score = 0.2  # Very short content
        elif combined_length < 100:
            length_score = 0.5  # Short content
        elif combined_length < 300:
            length_score = 0.8  # Medium content
        elif combined_length < 800:
            length_score = 1.0  # Good length
        else:
            length_score = 0.9  # Very long might be overwhelming
        
        quality_factors.append(length_score)
        
        # Factor 2: Information density (presence of technical terms, specifics)
        info_keywords = [
            'error', 'debug', 'fix', 'solution', 'example', 'code', 
            'function', 'method', 'api', 'database', 'query', 'response',
            'import', 'class', 'return', 'parameter', 'configuration'
        ]
        
        combined_text = (memory.user_message + " " + memory.ai_response).lower()
        keyword_matches = sum(1 for keyword in info_keywords if keyword in combined_text)
        info_density = min(1.0, keyword_matches / 5.0)  # Normalize to 0-1
        
        quality_factors.append(info_density)
        
        # Factor 3: Response completeness (AI gave substantial answer)
        ai_response_length = len(memory.ai_response)
        if ai_response_length < 20:
            completeness_score = 0.2  # Very short response
        elif ai_response_length < 100:
            completeness_score = 0.6  # Short response
        elif ai_response_length < 300:
            completeness_score = 1.0  # Good response
        else:
            completeness_score = 0.9  # Very long response
        
        quality_factors.append(completeness_score)
        
        # Factor 4: Question specificity (specific questions often have better answers)
        question_indicators = ['how', 'what', 'why', 'when', 'where', 'which']
        user_text_lower = memory.user_message.lower()
        has_question = any(indicator in user_text_lower for indicator in question_indicators)
        specificity_score = 0.8 if has_question else 0.6
        
        quality_factors.append(specificity_score)
        
        # Combine all quality factors with equal weighting
        overall_quality = sum(quality_factors) / len(quality_factors)
        return max(0.0, min(1.0, overall_quality))
    
    def calculate_context_score(self, memory: MessageMemory, query: str) -> float:
        """
        Calculate how well the memory fits the current conversation context.
        
        Analyzes topical alignment and conversation flow compatibility
        between the stored memory and current query.
        
        Args:
            memory: MessageMemory to assess for context fit
            query: Current search query for context comparison
            
        Returns:
            float: Context fit score (0.0-1.0)
        """
        context_factors = []
        
        # Factor 1: Query-memory topic alignment
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        memory_words = set(re.findall(
            r'\b\w+\b', 
            (memory.user_message + " " + memory.ai_response).lower()
        ))
        
        if query_words and memory_words:
            word_overlap = len(query_words.intersection(memory_words))
            word_alignment = word_overlap / len(query_words.union(memory_words))
            context_factors.append(word_alignment)
        else:
            context_factors.append(0.5)  # Neutral score if no words
        
        # Factor 2: Conversational continuity (question-answer patterns)
        query_is_question = any(q in query.lower() for q in ['?', 'how', 'what', 'why'])
        memory_has_answer = len(memory.ai_response) > len(memory.user_message)
        
        if query_is_question and memory_has_answer:
            continuity_score = 0.9  # Good question-answer pattern
        elif query_is_question:
            continuity_score = 0.7  # Question but weak answer
        else:
            continuity_score = 0.6  # Statement or general context
        
        context_factors.append(continuity_score)
        
        # Factor 3: Technical depth matching
        technical_terms = [
            'api', 'database', 'function', 'error', 'debug', 'code',
            'python', 'fastapi', 'sql', 'json', 'http', 'response'
        ]
        
        query_technical = sum(1 for term in technical_terms if term in query.lower())
        memory_technical = sum(1 for term in technical_terms 
                              if term in (memory.user_message + " " + memory.ai_response).lower())
        
        if query_technical > 0 and memory_technical > 0:
            tech_alignment = min(1.0, memory_technical / max(1, query_technical))
        else:
            tech_alignment = 0.5  # Neutral if no technical terms
        
        context_factors.append(tech_alignment)
        
        # Combine context factors
        overall_context = sum(context_factors) / len(context_factors)
        return max(0.0, min(1.0, overall_context))
    
    def calculate_final_score(self, memory: MessageMemory, query: str,
                             similarity: float) -> float:
        """
        Calculate final relevance score combining all factors.
        
        Args:
            memory: MessageMemory to score
            query: Search query for context
            similarity: Semantic similarity score
            
        Returns:
            float: Final relevance score (0.0-1.0)
        """
        # Calculate individual scores
        similarity_score = self.calculate_similarity_score(similarity)
        recency_score = self.calculate_recency_score(memory.timestamp)
        quality_score = self.calculate_quality_score(memory)
        context_score = self.calculate_context_score(memory, query)
        
        # Weight and combine scores
        final_score = (
            similarity_score * self.config.similarity_weight +
            recency_score * self.config.recency_weight +
            quality_score * self.config.quality_weight +
            context_score * self.config.context_weight
        )
        
        return max(0.0, min(1.0, final_score))


class QueryAnalyzer:
    """
    Intelligent query preprocessing and enhancement for better search results.
    
    Analyzes search queries to extract intent, expand terms, and optimize
    the search process for more relevant results.
    """
    
    def __init__(self, config: SearchConfig):
        """
        Initialize query analyzer with configuration.
        
        Args:
            config: SearchConfig with analysis parameters
        """
        self.config = config
        
        # Common term expansions for better matching
        self.term_expansions = {
            'debug': ['debugging', 'fix', 'error', 'troubleshoot'],
            'api': ['endpoint', 'route', 'service', 'interface'],
            'database': ['db', 'sql', 'query', 'table', 'schema'],
            'error': ['exception', 'bug', 'issue', 'problem', 'failure'],
            'test': ['testing', 'unittest', 'pytest', 'validation'],
            'deploy': ['deployment', 'production', 'server', 'hosting'],
            'config': ['configuration', 'settings', 'environment', 'setup']
        }
    
    def clean_query(self, query: str) -> str:
        """
        Clean and normalize the search query.
        
        Args:
            query: Raw search query
            
        Returns:
            str: Cleaned query
        """
        # Remove excessive whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', query.strip())
        
        # Remove special characters except question marks and hyphens
        cleaned = re.sub(r'[^\w\s\-\?]', ' ', cleaned)
        
        # Normalize to lowercase for analysis (but preserve original case)
        return cleaned
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract meaningful keywords from the query.
        
        Args:
            query: Search query
            
        Returns:
            List[str]: Important keywords
        """
        # Common stop words to filter out
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Extract words and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with related terms for better matching.
        
        Args:
            query: Original search query
            
        Returns:
            List[str]: Expanded query variations
        """
        if not self.config.enable_query_expansion:
            return [query]
        
        expanded_queries = [query]  # Always include original
        keywords = self.extract_keywords(query)
        
        # Generate expanded versions
        for keyword in keywords:
            if keyword in self.term_expansions:
                for expansion in self.term_expansions[keyword]:
                    # Create query variant with expanded term
                    expanded_query = query.replace(keyword, expansion)
                    if expanded_query != query:
                        expanded_queries.append(expanded_query)
        
        # Limit number of expansions
        return expanded_queries[:3]  # Original + up to 2 expansions
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine search intent and optimization hints.
        
        Args:
            query: Search query to analyze
            
        Returns:
            Dict[str, Any]: Analysis results with intent and hints
        """
        query_lower = query.lower()
        
        # Determine query type
        is_question = '?' in query or any(
            q in query_lower for q in ['how', 'what', 'why', 'when', 'where', 'which']
        )
        
        is_problem = any(
            p in query_lower for p in ['error', 'issue', 'problem', 'bug', 'fail', 'wrong']
        )
        
        is_tutorial = any(
            t in query_lower for t in ['tutorial', 'example', 'guide', 'learn', 'teach']
        )
        
        # Technical domain detection
        domains = {
            'web_development': ['api', 'endpoint', 'http', 'rest', 'fastapi', 'web', 'server'],
            'database': ['database', 'sql', 'query', 'table', 'db', 'schema'],
            'python': ['python', 'function', 'class', 'import', 'module', 'package'],
            'testing': ['test', 'testing', 'unittest', 'pytest', 'mock', 'validation'],
            'deployment': ['deploy', 'production', 'docker', 'server', 'host']
        }
        
        detected_domains = []
        for domain, keywords in domains.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_domains.append(domain)
        
        return {
            'is_question': is_question,
            'is_problem': is_problem,
            'is_tutorial': is_tutorial,
            'domains': detected_domains,
            'keywords': self.extract_keywords(query),
            'expanded_queries': self.expand_query(query)
        }


@dataclass
class SearchResult:
    """
    Enhanced search result with multi-factor relevance scoring.
    
    Contains the original memory plus detailed scoring breakdown
    and relevance metadata for transparency and debugging.
    
    Attributes:
        memory: Original MessageMemory from search
        relevance_score: Final combined relevance score (0.0-1.0)
        similarity_score: Semantic similarity component
        recency_score: Temporal relevance component  
        quality_score: Content quality component
        context_score: Conversation context fit component
        query_analysis: Analysis of the search query
        explanation: Human-readable relevance explanation
    """
    
    memory: MessageMemory
    relevance_score: float
    similarity_score: float
    recency_score: float
    quality_score: float
    context_score: float
    query_analysis: Dict[str, Any]
    explanation: str
    
    def get_score_breakdown(self) -> str:
        """
        Get human-readable breakdown of relevance scoring.
        
        Returns:
            str: Formatted score breakdown
        """
        return (
            f"Relevance: {self.relevance_score:.3f} "
            f"(Similarity: {self.similarity_score:.3f}, "
            f"Recency: {self.recency_score:.3f}, "
            f"Quality: {self.quality_score:.3f}, "
            f"Context: {self.context_score:.3f})"
        )


class SemanticSearchEngine:
    """
    Advanced semantic search engine with multi-factor relevance scoring.
    
    Provides enhanced search capabilities that go beyond basic similarity
    matching to include temporal relevance, content quality, and contextual fit.
    """
    
    def __init__(self, config: SearchConfig, memory_service: MemoryService):
        """
        Initialize semantic search engine.
        
        Args:
            config: SearchConfig with engine parameters
            memory_service: MemoryService for accessing stored memories
        """
        self.config = config
        self.memory_service = memory_service
        self.relevance_scorer = RelevanceScorer(config)
        self.query_analyzer = QueryAnalyzer(config)
        
        print("Semantic search engine initialized with multi-factor scoring")
    
    def search_memories(self, conversation_id: str, query: str,
                       max_results: Optional[int] = None,
                       include_context_analysis: bool = True) -> List[SearchResult]:
        """
        Perform advanced semantic search with relevance scoring.
        
        Args:
            conversation_id: ID of conversation to search within
            query: Search query text
            max_results: Maximum results to return (uses config default if None)
            include_context_analysis: Whether to include detailed query analysis
            
        Returns:
            List[SearchResult]: Enhanced search results ordered by relevance
            
        Example:
            results = search_engine.search_memories(
                conversation_id="conv_123",
                query="FastAPI debugging techniques",
                max_results=5,
                include_context_analysis=True
            )
            
            for result in results:
                print(f"Score: {result.relevance_score:.3f}")
                print(f"Memory: {result.memory.get_context_summary()}")
                print(f"Breakdown: {result.get_score_breakdown()}")
        """
        if not query.strip():
            return []
        
        max_results = max_results or self.config.max_search_results
        
        # Analyze query for optimization hints
        query_analysis = self.query_analyzer.analyze_query_intent(query) if include_context_analysis else {}
        
        print(f"Searching memories for: '{query}' in conversation {conversation_id[:8]}...")
        
        # Get base memories from memory service
        # Use slightly lower threshold to get more candidates for scoring
        base_threshold = max(0.5, self.config.similarity_weight * 0.7)
        
        base_memories = self.memory_service.get_relevant_memories(
            conversation_id=conversation_id,
            query_text=query,
            max_memories=max_results * 2,  # Get more candidates for reranking
            similarity_threshold=base_threshold
        )
        
        if not base_memories:
            print("No memories found matching query")
            return []
        
        # Enhanced scoring and reranking
        search_results = []
        
        for memory in base_memories:
            # Calculate comprehensive relevance score
            similarity = memory.similarity_score  # From original search
            
            similarity_score = self.relevance_scorer.calculate_similarity_score(similarity)
            recency_score = self.relevance_scorer.calculate_recency_score(memory.timestamp)
            quality_score = self.relevance_scorer.calculate_quality_score(memory)
            context_score = self.relevance_scorer.calculate_context_score(memory, query)
            
            final_score = self.relevance_scorer.calculate_final_score(memory, query, similarity)
            
            # Generate explanation
            explanation = self._generate_relevance_explanation(
                memory, query, similarity_score, recency_score, quality_score, context_score
            )
            
            # Create enhanced search result
            search_result = SearchResult(
                memory=memory,
                relevance_score=final_score,
                similarity_score=similarity_score,
                recency_score=recency_score,
                quality_score=quality_score,
                context_score=context_score,
                query_analysis=query_analysis,
                explanation=explanation
            )
            
            search_results.append(search_result)
        
        # Sort by final relevance score
        search_results.sort(key=lambda r: r.relevance_score, reverse=True)
        
        # Limit results
        final_results = search_results[:max_results]
        
        print(f"Enhanced search found {len(final_results)} relevant memories")
        for i, result in enumerate(final_results[:3]):  # Show top 3
            print(f"  {i+1}. {result.get_score_breakdown()}")
        
        return final_results
    
    def _generate_relevance_explanation(self, memory: MessageMemory, query: str,
                                      similarity: float, recency: float,
                                      quality: float, context: float) -> str:
        """
        Generate human-readable explanation for relevance scoring.
        
        Args:
            memory: The memory being scored
            query: Search query
            similarity: Similarity score
            recency: Recency score
            quality: Quality score
            context: Context score
            
        Returns:
            str: Relevance explanation
        """
        explanations = []
        
        # Similarity explanation
        if similarity > 0.8:
            explanations.append("highly relevant content")
        elif similarity > 0.6:
            explanations.append("moderately relevant content")
        else:
            explanations.append("somewhat related content")
        
        # Recency explanation
        hours_ago = (datetime.now() - memory.timestamp).total_seconds() / 3600
        if hours_ago < 24:
            explanations.append("recent discussion")
        elif hours_ago < 168:  # 1 week
            explanations.append("from this week")
        else:
            explanations.append("from earlier conversation")
        
        # Quality explanation
        if quality > 0.7:
            explanations.append("detailed information")
        elif quality > 0.5:
            explanations.append("useful content")
        else:
            explanations.append("basic information")
        
        # Context explanation
        if context > 0.7:
            explanations.append("good contextual fit")
        
        return ", ".join(explanations)
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about search engine configuration and performance.
        
        Returns:
            Dict[str, Any]: Search engine statistics
        """
        return {
            "scoring_weights": {
                "similarity": self.config.similarity_weight,
                "recency": self.config.recency_weight,
                "quality": self.config.quality_weight,
                "context": self.config.context_weight
            },
            "configuration": {
                "recency_decay_hours": self.config.recency_decay_hours,
                "max_results": self.config.max_search_results,
                "query_expansion": self.config.enable_query_expansion,
                "boost_threshold_hours": self.config.boost_recent_threshold_hours
            },
            "weights_valid": self.config.validate_weights()
        }


def init_search_engine(memory_service: MemoryService) -> SemanticSearchEngine:
    """
    Initialize semantic search engine with optimal configuration.
    
    Args:
        memory_service: Initialized MemoryService for accessing memories
        
    Returns:
        SemanticSearchEngine: Configured search engine
        
    Example:
        memory_service = init_memory_service()
        search_engine = init_search_engine(memory_service)
        
        results = search_engine.search_memories(
            conversation_id="conv_123",
            query="FastAPI debugging"
        )
    """
    try:
        # Create balanced search configuration
        search_config = SearchConfig(
            similarity_weight=0.4,    # Primary factor - semantic similarity
            recency_weight=0.3,       # Important - recent info more relevant  
            quality_weight=0.2,       # Moderate - content richness matters
            context_weight=0.1,       # Minor - conversational fit
            recency_decay_hours=168.0, # 1 week half-life
            min_content_length=20,
            max_search_results=10,
            enable_query_expansion=True,
            boost_recent_threshold_hours=24.0
        )
        
        # Validate configuration
        if not search_config.validate_weights():
            print("Warning: Search weights don't sum to 1.0")
        
        # Create search engine
        search_engine = SemanticSearchEngine(search_config, memory_service)
        
        print("Semantic search engine initialization completed")
        return search_engine
        
    except Exception as e:
        print(f"Failed to initialize semantic search engine: {e}")
        raise


def test_semantic_search() -> None:
    """
    Test semantic search engine with comprehensive scenarios.
    
    Validates enhanced search functionality including multi-factor scoring,
    query analysis, and relevance ranking with sample data.
    """
    print("\n🧪 Testing semantic search engine functionality...")
    
    try:
        # Test 1: Initialize services
        print("\n1. Initializing search engine and dependencies...")
        from services.memory_service import init_memory_service
        memory_service = init_memory_service()
        search_engine = init_search_engine(memory_service)
        
        # Test conversation ID
        test_conv_id = "test_search_conv_789"
        
        # Test 2: Store diverse sample memories
        print("\n2. Storing diverse sample memories...")
        
        sample_memories = [
            {
                "message_id": 201,
                "user_message": "How do I debug FastAPI applications effectively?",
                "ai_response": "FastAPI debugging involves several techniques: 1) Use logging with Python's logging module, 2) Enable debug mode in development, 3) Use breakpoints with pdb, 4) Check request/response data, 5) Monitor error traces.",
                "timestamp_offset_hours": 2  # 2 hours ago
            },
            {
                "message_id": 202, 
                "user_message": "What are FastAPI best practices for error handling?",
                "ai_response": "FastAPI error handling best practices include: custom exception handlers, proper HTTP status codes, structured error responses with detail messages, logging errors for monitoring, and graceful fallbacks.",
                "timestamp_offset_hours": 48  # 2 days ago
            },
            {
                "message_id": 203,
                "user_message": "How to optimize database queries in Python?",
                "ai_response": "Database optimization in Python: use connection pooling, implement query optimization, add proper indexes, use ORM efficiently with eager loading, and monitor query performance.",
                "timestamp_offset_hours": 168  # 1 week ago
            },
            {
                "message_id": 204,
                "user_message": "Simple greeting",
                "ai_response": "Hello!",
                "timestamp_offset_hours": 1  # 1 hour ago - low quality
            },
            {
                "message_id": 205,
                "user_message": "What is FastAPI middleware and how do I use it?",
                "ai_response": "FastAPI middleware runs before and after each request. You can create custom middleware for authentication, logging, CORS, or request processing. Use @app.middleware decorator or add_middleware method.",
                "timestamp_offset_hours": 24  # 1 day ago
            }
        ]
        
        # Store memories with different timestamps
        for memory in sample_memories:
            success = memory_service.store_message_memory(
                conversation_id=test_conv_id,
                message_id=memory["message_id"],
                user_message=memory["user_message"],
                ai_response=memory["ai_response"],
                immediate=True
            )
            print(f"Stored memory {memory['message_id']}: {'✅' if success else '❌'}")
        
        # Test 3: Basic semantic search
        print("\n3. Testing basic semantic search...")
        
        search_queries = [
            "FastAPI debugging techniques",
            "API error handling strategies", 
            "database performance optimization"
        ]
        
        for query in search_queries:
            results = search_engine.search_memories(
                conversation_id=test_conv_id,
                query=query,
                max_results=3
            )
            
            print(f"\nQuery: '{query}'")
            print(f"Found {len(results)} results:")
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.get_score_breakdown()}")
                print(f"     Memory: {result.memory.user_message[:60]}...")
                print(f"     Explanation: {result.explanation}")
        
        # Test 4: Query analysis
        print("\n4. Testing query analysis...")
        
        analyzer = QueryAnalyzer(SearchConfig())
        test_queries = [
            "How do I debug FastAPI errors?",
            "database optimization tutorial",
            "API testing best practices"
        ]
        
        for query in test_queries:
            analysis = analyzer.analyze_query_intent(query)
            print(f"\nQuery: '{query}'")
            print(f"  Type: {'Question' if analysis['is_question'] else 'Statement'}")
            print(f"  Domains: {', '.join(analysis['domains']) if analysis['domains'] else 'General'}")
            print(f"  Keywords: {', '.join(analysis['keywords'])}")
            print(f"  Expansions: {len(analysis['expanded_queries'])} variants")
        
        # Test 5: Relevance scoring components
        print("\n5. Testing relevance scoring components...")
        
        scorer = RelevanceScorer(SearchConfig())
        
        if sample_memories:
            # Test with first stored memory
            test_memory = MessageMemory(
                conversation_id=test_conv_id,
                message_id=201,
                user_message=sample_memories[0]["user_message"],
                ai_response=sample_memories[0]["ai_response"],
                combined_text=f"User: {sample_memories[0]['user_message']} | AI: {sample_memories[0]['ai_response']}",
                embedding_id="test_emb",
                similarity_score=0.85,
                timestamp=datetime.now() - timedelta(hours=2),
                metadata={}
            )
            
            query = "FastAPI debugging"
            
            similarity_score = scorer.calculate_similarity_score(0.85)
            recency_score = scorer.calculate_recency_score(test_memory.timestamp)
            quality_score = scorer.calculate_quality_score(test_memory)
            context_score = scorer.calculate_context_score(test_memory, query)
            final_score = scorer.calculate_final_score(test_memory, query, 0.85)
            
            print(f"Scoring breakdown for test memory:")
            print(f"  Similarity: {similarity_score:.3f}")
            print(f"  Recency: {recency_score:.3f}")
            print(f"  Quality: {quality_score:.3f}")
            print(f"  Context: {context_score:.3f}")
            print(f"  Final: {final_score:.3f}")
        
        # Test 6: Search engine statistics
        print("\n6. Getting search engine statistics...")
        stats = search_engine.get_search_statistics()
        print(f"Scoring weights: {stats['scoring_weights']}")
        print(f"Configuration valid: {'✅' if stats['weights_valid'] else '❌'}")
        
        print("\n✅ All semantic search engine tests passed!")
        
        # Cleanup
        memory_service.cleanup()
        
    except Exception as e:
        print(f"\n❌ Semantic search engine test failed: {e}")
        raise


# Development utility - run this file directly to test search engine
if __name__ == "__main__":
    """
    Test semantic search engine when run directly.
    
    Usage:
        python src/services/semantic_search.py
        
    This will test multi-factor relevance scoring, query analysis,
    and enhanced search capabilities with sample conversation data.
    """
    test_semantic_search()