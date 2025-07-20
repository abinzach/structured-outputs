"""Document Processing for handling large texts and chunking."""
import re
from typing import List, Dict, Any, Tuple
try:
    from utils import count_tokens, merge_dicts_deep
except ImportError:
    from .utils import count_tokens, merge_dicts_deep


class DocumentProcessor:
    """Handles document chunking and processing for large texts."""
    
    def __init__(self, max_tokens_per_chunk: int = 900000, overlap_tokens: int = 5000):
        # Updated for GPT-4.1's 1M token limit, leaving room for schema and prompt
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.overlap_tokens = overlap_tokens
    
    def process_document(self, text: str, schema: dict) -> Dict[str, Any]:
        """
        Process a document, chunking if necessary.
        
        Args:
            text: Input document text
            schema: JSON schema for extraction
            
        Returns:
            dict: Processing metadata and chunked text if applicable
        """
        token_count = count_tokens(text)
        
        if token_count <= self.max_tokens_per_chunk:
            return {
                'needs_chunking': False,
                'total_tokens': token_count,
                'chunks': [{'text': text, 'chunk_id': 0, 'start_pos': 0, 'end_pos': len(text)}],
                'total_chunks': 1
            }
        
        chunks = self.chunk_document(text)
        
        return {
            'needs_chunking': True,
            'total_tokens': token_count,
            'chunks': chunks,
            'total_chunks': len(chunks),
            'overlap_tokens': self.overlap_tokens
        }
    
    def chunk_document(self, text: str, max_tokens: int = None) -> List[Dict[str, Any]]:
        """
        Smart chunking with overlap and sentence boundary preservation.
        
        Args:
            text: Input text to chunk
            max_tokens: Maximum tokens per chunk (defaults to instance setting)
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if max_tokens is None:
            max_tokens = self.max_tokens_per_chunk
        
        # Split text into sentences for better boundary preservation
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_id = 0
        text_position = 0
        
        for sentence in sentences:
            sentence_tokens = count_tokens(sentence)
            
            # If single sentence exceeds max tokens, split it further
            if sentence_tokens > max_tokens:
                # Handle oversized sentences
                if current_chunk:
                    # Finalize current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'chunk_id': chunk_id,
                        'start_pos': text_position - len(chunk_text),
                        'end_pos': text_position,
                        'token_count': current_tokens,
                        'sentence_count': len(current_chunk)
                    })
                    chunk_id += 1
                    current_chunk = []
                    current_tokens = 0
                
                # Split oversized sentence
                sub_chunks = self._split_oversized_sentence(sentence, max_tokens)
                for sub_chunk in sub_chunks:
                    chunks.append({
                        'text': sub_chunk,
                        'chunk_id': chunk_id,
                        'start_pos': text_position,
                        'end_pos': text_position + len(sub_chunk),
                        'token_count': count_tokens(sub_chunk),
                        'sentence_count': 1
                    })
                    chunk_id += 1
                    text_position += len(sub_chunk)
                
                continue
            
            # Check if adding this sentence would exceed token limit
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                # Finalize current chunk with overlap
                chunk_text = ' '.join(current_chunk)
                
                # Add overlap from previous chunk if exists
                overlap_text = self._get_overlap_text(chunks, chunk_text)
                if overlap_text:
                    chunk_text = overlap_text + ' ' + chunk_text
                
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': chunk_id,
                    'start_pos': text_position - len(' '.join(current_chunk)),
                    'end_pos': text_position,
                    'token_count': count_tokens(chunk_text),
                    'sentence_count': len(current_chunk),
                    'has_overlap': bool(overlap_text)
                })
                
                chunk_id += 1
                
                # Start new chunk with overlap sentences
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_tokens = sum(count_tokens(s) for s in overlap_sentences)
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            text_position += len(sentence) + 1  # +1 for space
        
        # Add final chunk if not empty
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            
            # Add overlap from previous chunk if exists
            overlap_text = self._get_overlap_text(chunks, chunk_text)
            if overlap_text:
                chunk_text = overlap_text + ' ' + chunk_text
            
            chunks.append({
                'text': chunk_text,
                'chunk_id': chunk_id,
                'start_pos': text_position - len(' '.join(current_chunk)),
                'end_pos': text_position,
                'token_count': count_tokens(chunk_text),
                'sentence_count': len(current_chunk),
                'has_overlap': bool(overlap_text)
            })
        
        return chunks
    
    def summarize_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Create a coherent summary from document chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            str: Summary text for context
        """
        if not chunks:
            return ""
        
        if len(chunks) == 1:
            return chunks[0]['text'][:1000] + "..." if len(chunks[0]['text']) > 1000 else chunks[0]['text']
        
        # Extract key sentences from each chunk
        summary_parts = []
        
        for chunk in chunks[:5]:  # Limit to first 5 chunks for summary
            chunk_text = chunk['text']
            
            # Extract first and last sentences as key information
            sentences = self._split_into_sentences(chunk_text)
            
            if len(sentences) >= 2:
                summary_parts.append(sentences[0])
                if len(sentences) > 2:
                    summary_parts.append(sentences[-1])
            elif sentences:
                summary_parts.append(sentences[0])
        
        summary = ' '.join(summary_parts)
        
        # Truncate if too long
        if len(summary) > 2000:
            summary = summary[:2000] + "..."
        
        return summary
    
    def merge_extractions(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine results from multiple document chunks.
        
        Args:
            chunk_results: List of extraction results from each chunk
            
        Returns:
            dict: Merged extraction result
        """
        if not chunk_results:
            return {
                'data': {},
                'confidence': {'overall': 0.0, 'fields': {}},
                'strategy': 'document_chunked',
                'total_chunks': 0
            }
        
        if len(chunk_results) == 1:
            result = chunk_results[0].copy()
            result['strategy'] = 'document_chunked'
            result['total_chunks'] = 1
            return result
        
        # Merge data from all chunks
        merged_data = {}
        all_field_confidences = {}
        total_tokens = 0
        
        # Track field occurrences for confidence weighting
        field_occurrences = {}
        field_values = {}
        
        for chunk_result in chunk_results:
            chunk_data = chunk_result.get('data', {})
            chunk_confidence = chunk_result.get('confidence', {})
            
            # Merge data with conflict resolution
            merged_data = self._merge_with_conflict_resolution(merged_data, chunk_data, field_values, field_occurrences)
            
            # Aggregate confidence scores
            for field, confidence in chunk_confidence.get('fields', {}).items():
                if field not in all_field_confidences:
                    all_field_confidences[field] = []
                all_field_confidences[field].append(confidence)
            
            total_tokens += chunk_result.get('token_usage', 0)
        
        # Calculate weighted confidence scores
        final_confidences = {}
        for field, confidences in all_field_confidences.items():
            # Weight by frequency and average confidence
            avg_confidence = sum(confidences) / len(confidences)
            frequency_weight = len(confidences) / len(chunk_results)
            final_confidences[field] = avg_confidence * frequency_weight
        
        overall_confidence = sum(final_confidences.values()) / len(final_confidences) if final_confidences else 0.0
        
        return {
            'data': merged_data,
            'confidence': {
                'overall': overall_confidence,
                'fields': final_confidences
            },
            'strategy': 'document_chunked',
            'total_chunks': len(chunk_results),
            'token_usage': total_tokens,
            'chunk_results': chunk_results
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        # Enhanced sentence splitting pattern
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n+\s*(?=[A-Z])'
        
        sentences = re.split(sentence_pattern, text.strip())
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter out very short fragments
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences if cleaned_sentences else [text]
    
    def _split_oversized_sentence(self, sentence: str, max_tokens: int) -> List[str]:
        """Split an oversized sentence into smaller chunks."""
        # Try splitting by clauses first
        clause_patterns = [r',\s+', r';\s+', r'\s+and\s+', r'\s+or\s+', r'\s+but\s+']
        
        for pattern in clause_patterns:
            parts = re.split(pattern, sentence)
            if len(parts) > 1:
                chunks = []
                current_chunk = ""
                
                for part in parts:
                    test_chunk = current_chunk + part if current_chunk else part
                    if count_tokens(test_chunk) <= max_tokens:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = part
                
                if current_chunk:
                    chunks.append(current_chunk)
                
                if all(count_tokens(chunk) <= max_tokens for chunk in chunks):
                    return chunks
        
        # Fallback: split by words
        words = sentence.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            test_chunk = current_chunk + [word]
            if count_tokens(' '.join(test_chunk)) <= max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _get_overlap_text(self, existing_chunks: List[Dict[str, Any]], current_chunk_text: str) -> str:
        """Get overlap text from previous chunk."""
        if not existing_chunks:
            return ""
        
        last_chunk = existing_chunks[-1]
        last_chunk_sentences = self._split_into_sentences(last_chunk['text'])
        
        # Take last few sentences for overlap
        overlap_sentences = last_chunk_sentences[-2:] if len(last_chunk_sentences) >= 2 else last_chunk_sentences
        overlap_text = ' '.join(overlap_sentences)
        
        # Ensure overlap doesn't exceed token limit
        if count_tokens(overlap_text) > self.overlap_tokens:
            # Truncate to fit overlap token limit
            words = overlap_text.split()
            truncated = []
            current_tokens = 0
            
            for word in reversed(words):
                word_tokens = count_tokens(word)
                if current_tokens + word_tokens <= self.overlap_tokens:
                    truncated.insert(0, word)
                    current_tokens += word_tokens
                else:
                    break
            
            overlap_text = ' '.join(truncated)
        
        return overlap_text
    
    def _get_overlap_sentences(self, current_chunk: List[str]) -> List[str]:
        """Get sentences for overlap in next chunk."""
        if len(current_chunk) <= 2:
            return []
        
        # Take last 1-2 sentences for overlap
        overlap_count = min(2, len(current_chunk) // 3)
        return current_chunk[-overlap_count:] if overlap_count > 0 else []
    
    def _merge_with_conflict_resolution(self, merged_data: dict, chunk_data: dict, 
                                      field_values: dict, field_occurrences: dict) -> dict:
        """Merge chunk data with conflict resolution."""
        try:
            from utils import flatten_schema, get_nested_value, set_nested_value
        except ImportError:
            from .utils import flatten_schema, get_nested_value, set_nested_value
        
        # Flatten both dictionaries for easier processing
        merged_flat = self._flatten_dict(merged_data)
        chunk_flat = self._flatten_dict(chunk_data)
        
        # Process each field in chunk data
        for field_path, value in chunk_flat.items():
            if value is None:
                continue
            
            # Track field occurrences
            if field_path not in field_occurrences:
                field_occurrences[field_path] = 0
                field_values[field_path] = []
            
            field_occurrences[field_path] += 1
            field_values[field_path].append(value)
            
            # Conflict resolution strategy
            if field_path not in merged_flat:
                # New field, add it
                merged_flat[field_path] = value
            else:
                existing_value = merged_flat[field_path]
                
                # Resolve conflicts based on data type and frequency
                resolved_value = self._resolve_field_conflict(
                    existing_value, value, field_values[field_path]
                )
                merged_flat[field_path] = resolved_value
        
        # Reconstruct nested dictionary
        result = {}
        for field_path, value in merged_flat.items():
            set_nested_value(result, field_path, value)
        
        return result
    
    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '.') -> dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _resolve_field_conflict(self, existing_value: Any, new_value: Any, all_values: List[Any]) -> Any:
        """Resolve conflicts between field values from different chunks."""
        # If values are the same, no conflict
        if existing_value == new_value:
            return existing_value
        
        # For strings, prefer longer/more detailed value
        if isinstance(existing_value, str) and isinstance(new_value, str):
            return existing_value if len(existing_value) >= len(new_value) else new_value
        
        # For numbers, use the most frequent value
        if isinstance(existing_value, (int, float)) and isinstance(new_value, (int, float)):
            # Count occurrences
            value_counts = {}
            for val in all_values:
                value_counts[val] = value_counts.get(val, 0) + 1
            
            # Return most frequent value
            return max(value_counts.items(), key=lambda x: x[1])[0]
        
        # For lists, merge and deduplicate
        if isinstance(existing_value, list) and isinstance(new_value, list):
            combined = existing_value + new_value
            # Remove duplicates while preserving order
            seen = set()
            result = []
            for item in combined:
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            return result
        
        # For booleans, prefer True if any chunk has True
        if isinstance(existing_value, bool) and isinstance(new_value, bool):
            return existing_value or new_value
        
        # Default: prefer non-null, more recent value
        if new_value is not None:
            return new_value
        
        return existing_value
