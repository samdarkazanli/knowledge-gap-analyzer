RAG CORPUS FR — benchmark synthétique Elasticsearch / exploitation

Contenu :
- corpus/ : 30 documents à importer dans Open WebUI.
- benchmark/questions.txt : 10 questions de test.
- benchmark/answer-key.txt : vérité terrain. NE PAS IMPORTER dans Open WebUI.

Le corpus est entièrement synthétique. Il contient volontairement :
- rapports finaux et hypothèses préliminaires ;
- conversations Teams informelles ;
- runbooks actuels et documentation obsolète ;
- changements corrélés à des incidents ;
- distracteurs partageant les mêmes mots-clés ;
- identifiants exacts comme INC-001 et payments-write.

Protocole conseillé :
1. Créer une nouvelle Knowledge Collection Open WebUI.
2. Importer uniquement les 30 fichiers sous corpus/.
3. Garder Top K = 6.
4. Tester d'abord Hybrid Search ON, sans changer embeddings/chunking.
5. Poser les questions de benchmark dans de nouveaux chats.
6. Comparer les sources récupérées et les réponses à benchmark/answer-key.txt.
7. Répéter ensuite Vector-only / Hybrid / Hybrid+reranker et, plus tard,
   GPT-6 Luna / Qwen3.5-27B.

Important : ne jamais importer benchmark/ dans la Knowledge Collection.
