from typing import Dict, Optional, List, Tuple
from tqdm import tqdm


class TrieNode:
    __slots__ = ("count", "children", "char_sequence")
    def __init__(self):
        self.count = 0
        self.children: Dict[str, 'TrieNode'] = {}
        self.char_sequence: Optional[str] = None

def build_trie(texts: List[str], min_prefix_len=200, max_prefix_len=50000, step_size=8, threshold=20):
    """Optimized trie building with character-based processing"""
    root = TrieNode()

    for sample_idx, text in tqdm(enumerate(texts), desc="Building optimized trie"):
        if len(text) < min_prefix_len:
            continue
            
        # Process in chunks to reduce memory allocation
        node = root
        max_len = min(len(text), max_prefix_len)
        
        # First chunk: min_prefix_len
        chunk = text[:min_prefix_len]
        
        if chunk not in node.children:
            node.children[chunk] = TrieNode()
            node.children[chunk].char_sequence = chunk
        node = node.children[chunk]
        node.count += 1
        
        # Remaining chunks
        for i in range(min_prefix_len, max_len, step_size):
            end_idx = min(i + step_size, max_len)
            chunk = text[i:end_idx]
            
            if chunk not in node.children:
                node.children[chunk] = TrieNode()
                node.children[chunk].char_sequence = chunk
            node = node.children[chunk]
            node.count += 1
    
    return root

def collect_results(root, min_prefix_len=200, threshold=20):
    """Optimized result collection - iterative version"""
    results = []
    
    # Use a stack to replace recursion: (node, prefix_text)
    stack = [(root, "")]
    
    while stack:
        node, prefix_text = stack.pop()
        
        if node.count >= threshold or len(prefix_text) == 0:
            if len(prefix_text) >= min_prefix_len:
                # Skip if single child has same count (optimization from original)
                if not (len(node.children) == 1 and 
                       list(node.children.values())[0].count == node.count):
                    results.append((prefix_text, node.count))
        
        # Add children to stack for processing
        for child in node.children.values():
            if child.char_sequence is not None:
                new_prefix = prefix_text + child.char_sequence
                stack.append((child, new_prefix))
    
    return results

def collect_results_optimized(root, min_prefix_len=200, threshold=20):
    """Optimized result collection"""
    results = []
    
    def walk(node, prefix_text):
        if node.count >= threshold or len(prefix_text) == 0:
            if len(prefix_text) >= min_prefix_len:
                # Skip if single child has same count (optimization from original)
                if not (len(node.children) == 1 and 
                       list(node.children.values())[0].count == node.count):
                    results.append((prefix_text, node.count))
        
        for child in node.children.values():
            if child.char_sequence is not None:
                new_prefix = prefix_text + child.char_sequence
                walk(child, new_prefix)
    
    walk(root, "")
    return results

def get_frequent_prefixes(texts: List[str], min_prefix_len=200, max_prefix_len=50000, step_size=8, threshold=20, max_results=100) -> List[Tuple[str, int]]:
    """
    Find frequent prefixes in a list of text strings using character-based trie.
    
    Args:
        texts: List of text strings to analyze
        min_prefix_len: Minimum length of prefixes to consider (in characters)
        max_prefix_len: Maximum length of prefixes to consider (in characters)
        step_size: Step size for chunking during trie building
        threshold: Minimum frequency threshold for prefixes
        max_results: Maximum number of results to return
    
    Returns:
        List of tuples containing (prefix_text, frequency)
    """
    root = build_trie(texts, min_prefix_len, max_prefix_len, step_size, threshold)
    results = collect_results(root, min_prefix_len, threshold)

    results.sort(key=lambda x: x[1] * len(x[0]), reverse=True)
    results = results[:min(max_results, len(results))]
    
    results.sort(key=lambda x: len(x[0]), reverse=True) # sort by length to get the longest prefixes

    # filter out prefixes with length less than min_prefix_len
    results = [result for result in results if len(result[0]) >= max(min_prefix_len, 1)]

    return results