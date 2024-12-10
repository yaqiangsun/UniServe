# Qwen2-VL
Qwen2-vl multimodal model
## Docker
### Docker build
```bash
docker build -t qwen2vl .
```

### Docker run
```bash
docker run -d --name qwen2vl -p 8085:8085 qwen2vl 
```

### Request Test
```bash
python client.py --image test.png --prompt "testing prompt"
```