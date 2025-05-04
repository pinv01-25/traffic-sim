echo "Iniciando API en segundo plano..."
uvicorn server:app --host 0.0.0.0 --port 8001 --reload &
API_PID=$!

sleep 2

echo "Ejecutando simulación..."
python scripts/simulation/main.py

echo "Terminando API..."
kill $API_PID
