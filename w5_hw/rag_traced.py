# rag_traced.py
"""
RAGTraced - RAG with OpenTelemetry tracing
"""

from opentelemetry import trace
from rag_helper import RAGBase


class RAGTraced(RAGBase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_tokens = {}
    
    def rag(self, query):
        tracer = trace.get_tracer("llm-zoomcamp")
        with tracer.start_as_current_span("rag") as span:
            span.set_attribute("query", query)
            result = super().rag(query)
            span.set_attribute("result_length", len(result))
            return result
    
    def search(self, query, num_results=5):
        tracer = trace.get_tracer("llm-zoomcamp")
        with tracer.start_as_current_span("search") as span:
            span.set_attribute("query", query)
            span.set_attribute("num_results", num_results)
            results = super().search(query, num_results=num_results)
            span.set_attribute("actual_results", len(results))
            return results
    
    def llm(self, prompt):
        tracer = trace.get_tracer("llm-zoomcamp")
        with tracer.start_as_current_span("llm") as span:
            span.set_attribute("prompt_length", len(prompt))
            
            response = super().llm(prompt)

            print(f"Response type: {type(response)}")
            print(f"Has usage: {hasattr(response, 'usage')}")
            if hasattr(response, 'usage'):
                print(f"Usage: {response.usage}")

            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                total_tokens = response.usage.total_tokens
                
                self.last_tokens = {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens,
                }
                
                span.set_attribute("input_tokens", input_tokens)
                span.set_attribute("output_tokens", output_tokens)
                span.set_attribute("total_tokens", total_tokens)
                
                input_cost = (input_tokens / 1_000_000) * 0.150
                output_cost = (output_tokens / 1_000_000) * 0.600
                total_cost = input_cost + output_cost
                
                self.last_tokens['input_cost_usd'] = input_cost
                self.last_tokens['output_cost_usd'] = output_cost
                self.last_tokens['total_cost_usd'] = total_cost
                
                span.set_attribute("input_cost_usd", round(input_cost, 6))
                span.set_attribute("output_cost_usd", round(output_cost, 6))
                span.set_attribute("total_cost_usd", round(total_cost, 6))
            
            span.set_attribute("response_length", len(response.output_text))
            return response


