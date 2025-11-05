#!/usr/bin/env python3
"""
GPU Cluster Stress Test
Tests all 5 GPUs simultaneously to measure performance and token throughput
"""
import asyncio
import aiohttp
import time
import json
import subprocess
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any
import statistics

@dataclass
class ModelEndpoint:
    name: str
    url: str
    model_id: str
    gpus: str
    gpu_type: str

@dataclass
class TestResult:
    model: str
    prompt_length: int
    response_length: int
    total_duration: float
    tokens_per_second: float
    temperature: float
    max_tokens: int
    success: bool
    error: str = ""

class ClusterStressTester:
    def __init__(self):
        self.endpoints = [
            ModelEndpoint("Mistral-7B", "http://localhost:8000", "mistral:7b-instruct", "0,1", "NVIDIA"),
            ModelEndpoint("Llama-3.2-3B", "http://localhost:8001", "llama3.2:3b", "2,3", "NVIDIA"), 
            ModelEndpoint("Phi-3-Mini", "http://localhost:11434", "phi3:mini", "AMD", "AMD RX 6700 XT")
        ]
        
        self.test_prompts = {
            "short": "Explain AI in one sentence.",
            "medium": "Write a detailed explanation of machine learning algorithms and their applications in modern technology. Include examples.",
            "long": "Write a comprehensive essay about quantum computing, its principles, mathematical foundations, current applications, and future potential. Discuss quantum supremacy, quantum algorithms like Shor's and Grover's, and the challenges in building practical quantum computers. Include the latest developments in quantum error correction."
        }
        
        self.test_parameters = [
            {"temperature": 0.1, "max_tokens": 200},
            {"temperature": 0.7, "max_tokens": 400},  
            {"temperature": 1.0, "max_tokens": 600},
        ]
        
        self.results = []
        
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get current GPU statistics"""
        stats = {"timestamp": datetime.now().isoformat()}
        
        try:
            # NVIDIA GPU stats
            nvidia_result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw', '--format=csv,noheader,nounits'], 
                                         capture_output=True, text=True, timeout=10)
            if nvidia_result.returncode == 0:
                nvidia_lines = nvidia_result.stdout.strip().split('\n')
                stats['nvidia_gpus'] = []
                for line in nvidia_lines:
                    if line.strip():
                        parts = line.split(', ')
                        stats['nvidia_gpus'].append({
                            'index': int(parts[0]),
                            'name': parts[1],
                            'temp': float(parts[2]) if parts[2] != '[Not Supported]' else 0,
                            'utilization': float(parts[3]) if parts[3] != '[Not Supported]' else 0,
                            'memory_used': int(parts[4]),
                            'memory_total': int(parts[5]),
                            'power_draw': float(parts[6]) if parts[6] != '[Not Supported]' else 0
                        })
        except Exception as e:
            stats['nvidia_error'] = str(e)
            
        try:
            # AMD GPU stats
            rocm_result = subprocess.run(['rocm-smi', '--showtemp', '--showpower', '--showuse', '--showmemuse', '--csv'], 
                                       capture_output=True, text=True, timeout=10)
            if rocm_result.returncode == 0:
                lines = rocm_result.stdout.strip().split('\n')
                stats['amd_gpus'] = []
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        parts = line.split(',')
                        if len(parts) >= 6:
                            stats['amd_gpus'].append({
                                'index': parts[0],
                                'temp': parts[1],
                                'power': parts[2], 
                                'utilization': parts[3],
                                'memory_used': parts[4],
                                'memory_total': parts[5]
                            })
        except Exception as e:
            stats['amd_error'] = str(e)
            
        return stats

    async def run_inference(self, session: aiohttp.ClientSession, endpoint: ModelEndpoint, 
                          prompt: str, temperature: float, max_tokens: int) -> TestResult:
        """Run a single inference request"""
        payload = {
            "model": endpoint.model_id,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        }
        
        # Add max_tokens for Ollama
        if "max_tokens" in payload or True:  # Ollama uses different parameter names
            payload["options"] = {"num_predict": max_tokens, "temperature": temperature}
            
        start_time = time.time()
        
        try:
            async with session.post(f"{endpoint.url}/api/generate", 
                                  json=payload,
                                  timeout=aiohttp.ClientTimeout(total=120)) as response:
                
                if response.status == 200:
                    result = await response.json()
                    end_time = time.time()
                    
                    total_duration = end_time - start_time
                    response_text = result.get('response', '')
                    response_length = len(response_text.split())
                    
                    # Calculate tokens per second (approximate)
                    tokens_per_second = response_length / total_duration if total_duration > 0 else 0
                    
                    return TestResult(
                        model=endpoint.name,
                        prompt_length=len(prompt.split()),
                        response_length=response_length,
                        total_duration=total_duration,
                        tokens_per_second=tokens_per_second,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        success=True
                    )
                else:
                    return TestResult(
                        model=endpoint.name,
                        prompt_length=len(prompt.split()),
                        response_length=0,
                        total_duration=0,
                        tokens_per_second=0,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        success=False,
                        error=f"HTTP {response.status}"
                    )
                    
        except Exception as e:
            end_time = time.time()
            return TestResult(
                model=endpoint.name,
                prompt_length=len(prompt.split()),
                response_length=0,
                total_duration=end_time - start_time,
                tokens_per_second=0,
                temperature=temperature,
                max_tokens=max_tokens,
                success=False,
                error=str(e)
            )

    async def run_concurrent_test(self, prompt_type: str, temperature: float, max_tokens: int, 
                                iterations: int = 3) -> List[TestResult]:
        """Run concurrent tests on all endpoints"""
        prompt = self.test_prompts[prompt_type]
        print(f"\n🔥 Testing {prompt_type} prompt (temp={temperature}, max_tokens={max_tokens}) - {iterations} iterations")
        
        # Record GPU stats before test
        gpu_stats_before = self.get_gpu_stats()
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            # Create tasks for each endpoint and iteration
            for iteration in range(iterations):
                for endpoint in self.endpoints:
                    task = self.run_inference(session, endpoint, prompt, temperature, max_tokens)
                    tasks.append(task)
            
            # Run all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and convert to TestResult objects
            valid_results = []
            for result in results:
                if isinstance(result, TestResult):
                    valid_results.append(result)
                elif isinstance(result, Exception):
                    print(f"❌ Task failed: {result}")
            
        # Record GPU stats after test
        gpu_stats_after = self.get_gpu_stats()
        
        # Print immediate results
        for result in valid_results:
            status = "✅" if result.success else "❌"
            print(f"{status} {result.model}: {result.tokens_per_second:.2f} tokens/s "
                  f"({result.response_length} tokens in {result.total_duration:.2f}s)")
            
        return valid_results, gpu_stats_before, gpu_stats_after

    async def run_stress_test(self):
        """Run comprehensive stress test"""
        print("🚀 Starting GPU Cluster Stress Test")
        print("=" * 60)
        
        all_results = []
        all_gpu_stats = []
        
        # Test each combination of parameters
        for prompt_type in ["short", "medium", "long"]:
            for params in self.test_parameters:
                results, stats_before, stats_after = await self.run_concurrent_test(
                    prompt_type, 
                    params["temperature"], 
                    params["max_tokens"],
                    iterations=3
                )
                
                all_results.extend(results)
                all_gpu_stats.append({
                    'prompt_type': prompt_type,
                    'parameters': params,
                    'before': stats_before,
                    'after': stats_after
                })
                
                # Brief pause between test sets
                await asyncio.sleep(2)
        
        # Save detailed results
        self.save_results(all_results, all_gpu_stats)
        
        # Generate summary report
        self.generate_report(all_results)

    def save_results(self, results: List[TestResult], gpu_stats: List[Dict]):
        """Save detailed test results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results/cluster_stress_test_{timestamp}.json"
        
        # Convert results to dict format
        results_dict = []
        for result in results:
            results_dict.append({
                'model': result.model,
                'prompt_length': result.prompt_length,
                'response_length': result.response_length,
                'total_duration': result.total_duration,
                'tokens_per_second': result.tokens_per_second,
                'temperature': result.temperature,
                'max_tokens': result.max_tokens,
                'success': result.success,
                'error': result.error
            })
        
        data = {
            'timestamp': timestamp,
            'test_results': results_dict,
            'gpu_statistics': gpu_stats,
            'summary': self.calculate_summary_stats(results)
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📊 Detailed results saved to: {filename}")

    def calculate_summary_stats(self, results: List[TestResult]) -> Dict:
        """Calculate summary statistics"""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {"error": "No successful results"}
        
        # Group by model
        by_model = {}
        for result in successful_results:
            if result.model not in by_model:
                by_model[result.model] = []
            by_model[result.model].append(result.tokens_per_second)
        
        summary = {}
        total_tokens_per_second = 0
        
        for model, token_rates in by_model.items():
            avg_tokens_per_second = statistics.mean(token_rates)
            summary[model] = {
                'avg_tokens_per_second': avg_tokens_per_second,
                'min_tokens_per_second': min(token_rates),
                'max_tokens_per_second': max(token_rates),
                'std_dev': statistics.stdev(token_rates) if len(token_rates) > 1 else 0,
                'successful_requests': len(token_rates)
            }
            total_tokens_per_second += avg_tokens_per_second
        
        summary['cluster_total'] = {
            'total_tokens_per_second': total_tokens_per_second,
            'total_successful_requests': len(successful_results),
            'total_failed_requests': len(results) - len(successful_results),
            'success_rate': len(successful_results) / len(results) * 100
        }
        
        return summary

    def generate_report(self, results: List[TestResult]):
        """Generate and print performance report"""
        print("\n" + "=" * 60)
        print("📈 CLUSTER STRESS TEST RESULTS")
        print("=" * 60)
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Requests: {len(results)}")
        print(f"   Successful: {len(successful_results)} ({len(successful_results)/len(results)*100:.1f}%)")
        print(f"   Failed: {len(failed_results)} ({len(failed_results)/len(results)*100:.1f}%)")
        
        if successful_results:
            # Group by model
            by_model = {}
            for result in successful_results:
                if result.model not in by_model:
                    by_model[result.model] = []
                by_model[result.model].append(result)
            
            total_cluster_throughput = 0
            
            print(f"\n🎯 Per-Model Performance:")
            for model, model_results in by_model.items():
                token_rates = [r.tokens_per_second for r in model_results]
                avg_tokens_per_second = statistics.mean(token_rates)
                total_cluster_throughput += avg_tokens_per_second
                
                print(f"\n   {model}:")
                print(f"     Average: {avg_tokens_per_second:.2f} tokens/s")
                print(f"     Min/Max: {min(token_rates):.2f} - {max(token_rates):.2f} tokens/s")
                print(f"     Std Dev: {statistics.stdev(token_rates) if len(token_rates) > 1 else 0:.2f}")
                print(f"     Requests: {len(model_results)}")
            
            print(f"\n🚀 Cluster Total Throughput: {total_cluster_throughput:.2f} tokens/s")
            
            # Performance by prompt type
            by_prompt = {}
            for result in successful_results:
                # Estimate prompt type based on prompt length
                if result.prompt_length < 10:
                    prompt_type = "short"
                elif result.prompt_length < 30:
                    prompt_type = "medium" 
                else:
                    prompt_type = "long"
                    
                if prompt_type not in by_prompt:
                    by_prompt[prompt_type] = []
                by_prompt[prompt_type].append(result.tokens_per_second)
            
            print(f"\n📝 Performance by Prompt Type:")
            for prompt_type, rates in by_prompt.items():
                avg_rate = statistics.mean(rates)
                print(f"   {prompt_type.title()}: {avg_rate:.2f} tokens/s (avg)")

if __name__ == "__main__":
    tester = ClusterStressTester()
    asyncio.run(tester.run_stress_test())