# -*- coding: utf-8 -*-
import anthropic
import numpy as np
import pdfplumber
import os

# Get Claude API key with multiple fallback options
def get_claude_api_key():
    """Get Claude API key from multiple sources with fallbacks"""
    # First try environment variable (for production/DO deployment)
    api_key = os.getenv('CLAUDE_API_KEY')
    if api_key:
        print("✓ Using Claude API key from environment variable")
        return api_key
    
    # Try to import from local config
    try:
        from config_local import CLAUDE_API_KEY
        if CLAUDE_API_KEY and CLAUDE_API_KEY != "your-claude-api-key-here":
            print("✓ Using Claude API key from config_local.py")
            return CLAUDE_API_KEY
    except ImportError:
        pass
    
    # Try to import from main config
    try:
        from config import CLAUDE_API_KEY
        if CLAUDE_API_KEY and CLAUDE_API_KEY != "your-claude-api-key-here":
            print("✓ Using Claude API key from config.py")
            return CLAUDE_API_KEY
    except ImportError:
        pass
    
    # If no key found, return empty string
    print("⚠️ No Claude API key found! Check environment variables or config files.")
    return ""

# Get the API key
CLAUDE_API_KEY = get_claude_api_key()

# Initialize Anthropic client with error handling
def get_anthropic_client():
    """Get Anthropic client with proper error handling"""
    if not CLAUDE_API_KEY:
        raise ValueError("Claude API key not found. Please set CLAUDE_API_KEY environment variable.")
    return anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# Initialize client
try:
    client = get_anthropic_client()
    print("✓ Anthropic client initialized successfully")
except Exception as e:
    print(f"⚠️ Failed to initialize Anthropic client: {e}")
    client = None

def read_ieee_paper():
    """Read and extract text from the IEEE paper PDF"""
    try:
        pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "IEEEsub.pdf")
        if not os.path.exists(pdf_path):
            return "IEEE paper not found at expected location."
        
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        
        return text if text.strip() else "Could not extract text from IEEE paper."
    except Exception as e:
        return f"Error reading IEEE paper: {str(e)}"

# Cache the IEEE paper content
ieee_paper_content = read_ieee_paper()

async def claude_agent(user_message: str, signal_data: dict = None):
    """
    Claude AI Agent that provides expert analysis and insights about the Coherent Multiplex signal analysis system.
    """
    
    # System prompt with essential knowledge about the app
    system_prompt = """You are Claude 3 Haiku, an AI agent for the Coherent Multiplex signal analysis system.

## MODEL CONTEXT:
You are running on Claude 3 Haiku (claude-3-haiku-20240307), optimized for speed and cost-efficiency. Your responses are limited to 1000 tokens, so be concise while maintaining technical accuracy.

## INTELLECTUAL PROPERTY:
The Coherent Multiplex mathematical architecture has a provisional patent filed in the United States. When asked about intellectual property, licensing, or commercial use, acknowledge this patent protection and suggest contacting the patent holder for licensing inquiries.

## SYSTEM OVERVIEW:
The Coherent Multiplex analyzes 8 synthetic signals (A-H) at 100 Hz sampling rate using:
- Real-time FFT analysis
- Cosine similarity calculations between signals
- Wavelet coherence computation for signal pairs
- Interactive visualization of signals, spectra, and coherence

## KEY FEATURES:
- 8 signals with mixed sine waves (5-50 Hz range)
- Rolling buffer (256 points) for real-time processing
- Wavelet coherence using CWT (2-50 Hz, 200 frequency bins)
- Distance metrics: cosine similarity between FFT outputs
- Network visualization showing signal relationships

## YOUR ROLE:
- Analyze signal patterns and coherence data
- Explain signal processing concepts clearly
- Identify anomalies or interesting features
- Provide insights about signal relationships
- Answer questions about system functionality

## ANALYSIS FOCUS:
- Signal statistics (mean, RMS, energy)
- Frequency domain characteristics
- Coherence strength and phase relationships
- Similarity patterns between signals
- Anomalies or artifacts

## RESPONSE FORMAT:
- Use numbered lists for main points
- Use bullet points for sub-items
- Include line breaks between sections
- Use LaTeX for math: \\(inline\\) or \\[display\\]
- Keep responses concise and focused (1000 token limit)

## MATH FORMULAS:
- Wavelet coherence: \\(R^2(f,t) = \\frac{|S(W_{XY})|^2}{S(|W_X|^2) \\cdot S(|W_Y|^2)}\\)
- Cosine similarity: \\(\\text{sim} = \\frac{\\vec{a} \\cdot \\vec{b}}{||\\vec{a}|| \\cdot ||\\vec{b}||}\\)"""

    # Prepare the user message with signal data context
    user_context = f"User Question: {user_message}\n\n"
    
    # Check if user is asking about the IEEE paper or mathematical details
    paper_keywords = ['ieee', 'paper', 'mathematical', 'architecture', 'theory', 'formula', 'equation', 'definition']
    if any(keyword in user_message.lower() for keyword in paper_keywords):
        user_context += f"""
=== IEEE PAPER REFERENCE ===
The user is asking about mathematical or theoretical aspects. Here's the relevant content from IEEEsub.pdf:

{ieee_paper_content[:3000]}...

(Note: Content truncated for context limits. Full paper contains complete mathematical definitions and proofs.)

===========================

"""
    
    if signal_data:
        # Extract comprehensive information from signal data
        signals = signal_data.get('signals', [])
        fft_data = signal_data.get('fft', [])
        distances = signal_data.get('distances', [])
        coherence_info = signal_data.get('wavelet_coherence')
        
        # Analyze signal characteristics
        signal_stats = []
        if signals and len(signals) >= 8:
            for i, signal in enumerate(signals):
                if signal:
                    signal_array = np.array(signal)
                    stats = {
                        'label': chr(65 + i),  # A, B, C, etc.
                        'mean': float(np.mean(signal_array)),
                        'std': float(np.std(signal_array)),
                        'rms': float(np.sqrt(np.mean(signal_array**2))),
                        'min': float(np.min(signal_array)),
                        'max': float(np.max(signal_array)),
                        'energy': float(np.sum(signal_array**2))
                    }
                    signal_stats.append(stats)
        
        # Analyze frequency domain
        freq_analysis = []
        if fft_data and len(fft_data) >= 8:
            for i, fft_signal in enumerate(fft_data):
                if fft_signal:
                    fft_array = np.array(fft_signal)
                    # Find dominant frequencies (peaks)
                    freqs = np.fft.fftfreq(len(fft_array), 1/100)[:len(fft_array)//2]  # FS = 100
                    magnitude = fft_array[:len(fft_array)//2]
                    
                    # Find top 3 frequency peaks
                    peak_indices = np.argsort(magnitude)[-3:][::-1]
                    dominant_freqs = [{'freq': float(freqs[idx]), 'magnitude': float(magnitude[idx])} 
                                    for idx in peak_indices if freqs[idx] > 0]
                    
                    freq_stats = {
                        'label': chr(65 + i),
                        'dominant_frequencies': dominant_freqs,
                        'spectral_centroid': float(np.sum(freqs * magnitude) / np.sum(magnitude)) if np.sum(magnitude) > 0 else 0,
                        'spectral_bandwidth': float(np.sqrt(np.sum(((freqs - np.sum(freqs * magnitude) / np.sum(magnitude))**2) * magnitude) / np.sum(magnitude))) if np.sum(magnitude) > 0 else 0,
                        'total_power': float(np.sum(magnitude**2))
                    }
                    freq_analysis.append(freq_stats)
        
        # Analyze similarity matrix
        similarity_analysis = ""
        if distances:
            similarities = [1 - d for d in distances]  # Convert distances back to similarities
            avg_similarity = np.mean(similarities)
            max_similarity = np.max(similarities)
            min_similarity = np.min(similarities)
            
            # Find most and least similar pairs
            pair_labels = []
            for i in range(8):
                for j in range(i + 1, 8):
                    pair_labels.append((chr(65 + i), chr(65 + j)))
            
            max_idx = np.argmax(similarities)
            min_idx = np.argmin(similarities)
            
            similarity_analysis = f"""
Similarity Network Analysis:
- Average similarity: {avg_similarity:.3f}
- Most similar pair: {pair_labels[max_idx][0]}-{pair_labels[max_idx][1]} (similarity: {max_similarity:.3f})
- Least similar pair: {pair_labels[min_idx][0]}-{pair_labels[min_idx][1]} (similarity: {min_similarity:.3f})
- Similarity range: {min_similarity:.3f} to {max_similarity:.3f}
"""
        
        # Detailed coherence analysis
        coherence_analysis = ""
        if coherence_info:
            pair = coherence_info.get('pair', [])
            similarity = coherence_info.get('similarity', 0)
            pair_labels = coherence_info.get('pair_labels', [])
            coherence_matrix = coherence_info.get('coherence', [])
            phases = coherence_info.get('phases', [])
            freqs = coherence_info.get('freqs', [])
            
            if coherence_matrix and freqs:
                # Analyze coherence patterns
                coh_array = np.array(coherence_matrix)
                phase_array = np.array(phases) if phases else None
                freq_array = np.array(freqs)
                
                # Find frequency bands with high coherence
                if coh_array.ndim == 2:
                    avg_coherence_per_freq = np.mean(coh_array, axis=1)
                    max_coherence_freq_idx = np.argmax(avg_coherence_per_freq)
                    max_coherence_freq = freq_array[max_coherence_freq_idx]
                    max_coherence_value = avg_coherence_per_freq[max_coherence_freq_idx]
                    
                    # Find high coherence regions (>0.7)
                    high_coh_mask = coh_array > 0.7
                    high_coh_freqs = []
                    if np.any(high_coh_mask):
                        freq_indices = np.where(np.any(high_coh_mask, axis=1))[0]
                        high_coh_freqs = [float(freq_array[i]) for i in freq_indices]
                    
                    # Phase analysis
                    phase_analysis = ""
                    if phase_array is not None and phase_array.size > 0:
                        avg_phase = np.mean(phase_array)
                        phase_std = np.std(phase_array)
                        phase_analysis = f"""
Phase Relationship:
- Average phase difference: {avg_phase:.3f} radians ({np.degrees(avg_phase):.1f}°)
- Phase stability (std): {phase_std:.3f} radians
- Phase coupling: {'Strong' if phase_std < 0.5 else 'Moderate' if phase_std < 1.0 else 'Weak'}
"""
                    
                    coherence_analysis = f"""
Wavelet Coherence Analysis:
- Signal pair under analysis: {pair_labels[0] if len(pair_labels) > 0 else 'N/A'} & {pair_labels[1] if len(pair_labels) > 1 else 'N/A'}
- Pair similarity (cosine): {similarity:.4f}
- Peak coherence frequency: {max_coherence_freq:.2f} Hz (coherence: {max_coherence_value:.3f})
- High coherence frequencies (>0.7): {high_coh_freqs[:5] if high_coh_freqs else 'None'}
- Overall coherence strength: {'Strong' if max_coherence_value > 0.8 else 'Moderate' if max_coherence_value > 0.5 else 'Weak'}
{phase_analysis}
"""
        
        # Create frequency analysis strings without nested f-strings
        freq_lines = []
        for f_data in freq_analysis:
            freq_str_parts = []
            for df in f_data['dominant_frequencies'][:2]:
                freq_str_parts.append(f"{df['freq']:.1f}Hz({df['magnitude']:.2f})")
            freq_line = f"Signal {f_data['label']}: Dominant freqs: {', '.join(freq_str_parts)}, Centroid: {f_data['spectral_centroid']:.1f}Hz"
            freq_lines.append(freq_line)
        
        user_context += f"""
=== REAL-TIME SIGNAL ANALYSIS ===

TIME DOMAIN ANALYSIS:
{chr(10).join([f"Signal {s['label']}: Mean={s['mean']:.3f}, RMS={s['rms']:.3f}, Energy={s['energy']:.1f}" for s in signal_stats[:4]])}
{chr(10).join([f"Signal {s['label']}: Mean={s['mean']:.3f}, RMS={s['rms']:.3f}, Energy={s['energy']:.1f}" for s in signal_stats[4:]])}

FREQUENCY DOMAIN ANALYSIS:
{chr(10).join(freq_lines[:4])}
{chr(10).join(freq_lines[4:])}

{similarity_analysis}
{coherence_analysis}

Current Data Summary:
- Number of signals: {len(signal_data.get('signals', []))}
- FFT data available: {'Yes' if signal_data.get('fft') else 'No'}
- Coherence analysis: {'Available' if coherence_info else 'Pending'}
- Timestamp: {signal_data.get('timestamp', 'N/A')}

"""
    else:
        user_context += "Current Signal Analysis: No signal data available for analysis.\n\n"
    
    try:
        # Check if client is properly initialized
        if client is None:
            return {
                "response": "Claude API client not initialized. Please check your API key configuration.",
                "status": "error"
            }
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.7,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_context
                }
            ]
        )
        
        return {
            "response": response.content[0].text,
            "status": "success"
        }
        
    except ValueError as ve:
        return {
            "response": f"Configuration error: {str(ve)}",
            "status": "error"
        }
    except anthropic.APIConnectionError as e:
        return {
            "response": f"Failed to connect to Claude API: {str(e)}",
            "status": "error"
        }
    except anthropic.AuthenticationError as e:
        return {
            "response": f"Claude API authentication failed - check your API key: {str(e)}",
            "status": "error"
        }
    except anthropic.RateLimitError as e:
        return {
            "response": f"Claude API rate limit exceeded: {str(e)}",
            "status": "error"
        }
    except Exception as e:
        return {
            "response": f"Error communicating with Claude: {str(e)}",
            "status": "error"
        }

def get_ieee_paper_content():
    """Get the IEEE paper content"""
    return ieee_paper_content
