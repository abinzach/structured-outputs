"""Schema Analysis Engine for complex JSON schema handling."""
import json
from typing import Dict, List, Tuple, Any
try:
    from utils import (
        estimate_schema_tokens, calculate_schema_depth, count_schema_objects,
        count_enum_values, flatten_schema
    )
except ImportError:
    from .utils import (
        estimate_schema_tokens, calculate_schema_depth, count_schema_objects,
        count_enum_values, flatten_schema
    )


class SchemaAnalyzer:
    """Analyzes JSON schemas and provides extraction strategies."""
    
    def __init__(self, max_tokens_per_chunk: int = 100000):
        # Updated for GPT-4.1's 1M token limit - schemas can be much larger now
        self.max_tokens_per_chunk = max_tokens_per_chunk
    
    def analyze_complexity(self, schema: dict) -> dict:
        """
        Calculate schema metrics and complexity score.
        
        Returns:
            dict: Complexity analysis with metrics and strategy recommendation
        """
        metrics = {
            'max_depth': calculate_schema_depth(schema),
            'total_fields': len(flatten_schema(schema)),
            'object_count': count_schema_objects(schema),
            'enum_complexity': count_enum_values(schema),
            'estimated_tokens': estimate_schema_tokens(schema),
            'required_fields': self._count_required_fields(schema)
        }
        
        # Calculate complexity score (0-100)
        complexity_score = self._calculate_complexity_score(metrics)
        
        # Determine extraction strategy
        strategy = self._determine_strategy(metrics)
        
        return {
            'metrics': metrics,
            'complexity_score': complexity_score,
            'strategy': strategy,
            'needs_chunking': metrics['estimated_tokens'] > self.max_tokens_per_chunk
        }
    
    def build_dependency_graph(self, schema: dict) -> dict:
        """
        Build a dependency graph for schema fields.
        
        Returns:
            dict: Dependency mapping for ordered extraction
        """
        dependencies = {}
        flattened = flatten_schema(schema)
        
        for field_path, field_schema in flattened.items():
            dependencies[field_path] = {
                'depends_on': self._find_dependencies(field_path, field_schema, flattened),
                'required': field_schema.get('required', False),
                'type': field_schema.get('type', 'unknown'),
                'has_enum': 'enum' in field_schema
            }
        
        return dependencies
    
    def chunk_schema(self, schema: dict, max_tokens: int = None) -> List[dict]:
        """
        Break large schemas into manageable chunks while preserving relationships.
        
        Args:
            schema: The JSON schema to chunk
            max_tokens: Maximum tokens per chunk (defaults to instance setting)
            
        Returns:
            List of schema chunks with metadata
        """
        if max_tokens is None:
            max_tokens = self.max_tokens_per_chunk
        
        estimated_tokens = estimate_schema_tokens(schema)
        
        if estimated_tokens <= max_tokens:
            return [{
                'schema': schema,
                'chunk_id': 0,
                'total_chunks': 1,
                'dependencies': [],
                'priority': 'high'
            }]
        
        # Build dependency graph for intelligent chunking
        dependencies = self.build_dependency_graph(schema)
        
        # Group fields by dependency levels
        chunks = self._create_dependency_chunks(schema, dependencies, max_tokens)
        
        return chunks
    
    def get_extraction_order(self, schema: dict) -> List[str]:
        """
        Determine optimal field extraction order based on dependencies.
        
        Returns:
            List of field paths in extraction order
        """
        dependencies = self.build_dependency_graph(schema)
        
        # Topological sort based on dependencies
        ordered_fields = []
        visited = set()
        temp_visited = set()
        
        def visit(field):
            if field in temp_visited:
                # Circular dependency - add to end
                return
            if field in visited:
                return
            
            temp_visited.add(field)
            
            # Visit dependencies first
            for dep in dependencies.get(field, {}).get('depends_on', []):
                if dep in dependencies:
                    visit(dep)
            
            temp_visited.remove(field)
            visited.add(field)
            ordered_fields.append(field)
        
        for field in dependencies:
            if field not in visited:
                visit(field)
        
        return ordered_fields
    
    def _calculate_complexity_score(self, metrics: dict) -> int:
        """Calculate complexity score from 0-100."""
        score = 0
        
        # Depth contribution (0-30 points)
        depth_score = min(30, metrics['max_depth'] * 5)
        score += depth_score
        
        # Field count contribution (0-25 points)
        field_score = min(25, metrics['total_fields'] / 10)
        score += field_score
        
        # Object count contribution (0-25 points)
        object_score = min(25, metrics['object_count'] * 2)
        score += object_score
        
        # Enum complexity contribution (0-20 points)
        enum_score = min(20, metrics['enum_complexity'] / 50)
        score += enum_score
        
        return int(score)
    
    def _determine_strategy(self, metrics: dict) -> str:
        """Determine extraction strategy based on metrics."""
        if metrics['estimated_tokens'] > self.max_tokens_per_chunk:
            return 'multi_pass_chunked'
        elif metrics['max_depth'] > 1 or metrics['object_count'] > 1:
            return 'multi_pass_hierarchical'
        else:
            return 'single_pass'
    
    def _count_required_fields(self, schema: dict, path: str = "") -> int:
        """Count required fields in schema."""
        count = 0
        
        if 'required' in schema and isinstance(schema['required'], list):
            count += len(schema['required'])
        
        if 'properties' in schema:
            for key, value in schema['properties'].items():
                count += self._count_required_fields(value, f"{path}.{key}" if path else key)
        
        return count
    
    def _find_dependencies(self, field_path: str, field_schema: dict, all_fields: dict) -> List[str]:
        """Find dependencies for a field based on schema references."""
        dependencies = []
        
        # Check for $ref dependencies
        if '$ref' in field_schema:
            ref_path = field_schema['$ref'].replace('#/', '').replace('/', '.')
            if ref_path in all_fields:
                dependencies.append(ref_path)
        
        # Check for conditional dependencies (if/then/else)
        if 'if' in field_schema:
            # Fields that this field depends on for conditional logic
            if_schema = field_schema['if']
            dependencies.extend(self._extract_field_references(if_schema))
        
        return dependencies
    
    def _extract_field_references(self, schema_part: dict) -> List[str]:
        """Extract field references from schema part."""
        references = []
        
        if isinstance(schema_part, dict):
            for key, value in schema_part.items():
                if key == 'properties' and isinstance(value, dict):
                    references.extend(value.keys())
                elif isinstance(value, (dict, list)):
                    references.extend(self._extract_field_references(value))
        elif isinstance(schema_part, list):
            for item in schema_part:
                references.extend(self._extract_field_references(item))
        
        return references
    
    def _create_dependency_chunks(self, schema: dict, dependencies: dict, max_tokens: int) -> List[dict]:
        """Create schema chunks based on dependency analysis."""
        chunks = []
        current_chunk = {'properties': {}}
        current_tokens = 0
        chunk_id = 0
        
        # Sort fields by dependency level and importance
        ordered_fields = self.get_extraction_order(schema)
        
        for field_path in ordered_fields:
            field_parts = field_path.split('.')
            field_schema = self._get_field_schema(schema, field_parts)
            field_tokens = estimate_schema_tokens({field_path: field_schema})
            
            # Check if adding this field would exceed token limit
            if current_tokens + field_tokens > max_tokens and current_chunk['properties']:
                # Finalize current chunk
                chunks.append({
                    'schema': current_chunk,
                    'chunk_id': chunk_id,
                    'total_chunks': 0,  # Will be updated later
                    'dependencies': self._get_chunk_dependencies(current_chunk, dependencies),
                    'priority': self._calculate_chunk_priority(current_chunk, dependencies)
                })
                
                # Start new chunk
                current_chunk = {'properties': {}}
                current_tokens = 0
                chunk_id += 1
            
            # Add field to current chunk
            self._add_field_to_chunk(current_chunk, field_parts, field_schema)
            current_tokens += field_tokens
        
        # Add final chunk if not empty
        if current_chunk['properties']:
            chunks.append({
                'schema': current_chunk,
                'chunk_id': chunk_id,
                'total_chunks': 0,
                'dependencies': self._get_chunk_dependencies(current_chunk, dependencies),
                'priority': self._calculate_chunk_priority(current_chunk, dependencies)
            })
        
        # Update total_chunks count
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total_chunks
        
        return chunks
    
    def _get_field_schema(self, schema: dict, field_parts: List[str]) -> dict:
        """Get schema for a specific field path."""
        current = schema
        
        for part in field_parts:
            if 'properties' in current and part in current['properties']:
                current = current['properties'][part]
            else:
                return {}
        
        return current
    
    def _add_field_to_chunk(self, chunk: dict, field_parts: List[str], field_schema: dict):
        """Add a field to a chunk schema."""
        current = chunk
        
        for part in field_parts[:-1]:
            if 'properties' not in current:
                current['properties'] = {}
            if part not in current['properties']:
                current['properties'][part] = {'type': 'object', 'properties': {}}
            current = current['properties'][part]
        
        if 'properties' not in current:
            current['properties'] = {}
        current['properties'][field_parts[-1]] = field_schema
    
    def _get_chunk_dependencies(self, chunk: dict, dependencies: dict) -> List[str]:
        """Get dependencies for a chunk."""
        chunk_fields = set(flatten_schema(chunk).keys())
        chunk_deps = set()
        
        for field in chunk_fields:
            if field in dependencies:
                for dep in dependencies[field].get('depends_on', []):
                    if dep not in chunk_fields:
                        chunk_deps.add(dep)
        
        return list(chunk_deps)
    
    def _calculate_chunk_priority(self, chunk: dict, dependencies: dict) -> str:
        """Calculate priority for a chunk."""
        chunk_fields = flatten_schema(chunk)
        required_count = sum(1 for field, schema in chunk_fields.items() 
                           if dependencies.get(field, {}).get('required', False))
        
        if required_count > len(chunk_fields) * 0.7:
            return 'high'
        elif required_count > len(chunk_fields) * 0.3:
            return 'medium'
        else:
            return 'low'
