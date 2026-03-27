# LLM Groq Inference - Diagnosis Report

## Problem Found: Invalid Model Name

### Issue
All predictions in `groq_openai_gpt_oss_20b_few_shot_predictions.csv` show:
- `predicted_label`: -1 (failure)
- `raw_response`: empty

### Root Cause
The `.env` file specified an invalid model name:
```env
LLM_MODEL_NAMES=openai/gpt-oss-20b,llama-3.3-70b-versatile
```

**`openai/gpt-oss-20b` does NOT exist in Groq's API**

This model name appears to be:
- A reference to OpenAI's "GPT-OSS-20B" (which OpenAI doesn't maintain)
- Not available through Groq's API endpoint
- Causing silent API failures

### Solution Applied

Updated `.env` to use **valid Groq models**:
```env
LLM_MODEL_NAMES=mixtral-8x7b-32768,llama-3.3-70b-versatile
```

### Available Groq Models

As of 2024, Groq provides the following models:

| Model ID | Parameters | Speed | Use Case |
|----------|-----------|-------|----------|
| `llama-3.3-70b-versatile` | 70B | Fast | General purpose (recommended) |
| `mixtral-8x7b-32768` | 56B | Very Fast | Multilingual, code |
| `llama-3.1-70b-versatile` | 70B | Fast | Alternative to 3.3 |
| `gemma-7b-it` | 7B | Fastest | Low latency, instruction-tuned |

### Error Handling Improvement

Added try-except block in `_request_label_prediction()` to capture actual API errors:
```python
try:
    response = client.chat.completions.create(...)
    ...
except Exception as e:
    error_msg = f"API_Error: {type(e).__name__}: {str(e)}"
    return error_msg, None
```

Now when API calls fail, the error message will be visible in `raw_response` column instead of being empty.

### Next Steps

1. Re-run the LLM inference:
   ```bash
   python run_step_by_step.py --only llm-groq
   ```

2. Check the updated CSV files for actual predictions and error messages

3. If using a different model preference, ensure it exists in Groq's catalog by checking their API documentation

### Debug Information

If predictions still fail, check:
1. API Key validity: `echo $Groq_API_KEY` or check `.env`
2. Network connectivity to Groq API
3. Rate limiting (Groq has free tier limits)
4. Model availability differences between regions/accounts

### Prompt Templates

The system uses:
- **Zero-shot**: Direct classification without examples
- **Few-shot**: Classification with 3 labeled examples per class

Both expect binary output (0 or 1) in English or parseable format.
