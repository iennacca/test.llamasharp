namespace test.llamasharp.Samples 
{
    using LLama;
    using LLama.Common;

    internal class ChatBot 
    {
        public static async Task Run()
        {
            // change it to your own model path
            var modelPath = @"D:\Users\Jerry\Projects\common\models\llama-2-7b-chat.Q2_K.gguf"; 
            // use the "chat-with-bob" prompt here.
            var prompt = @"Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.\r\n\r\nUser: Hello, Bob.\r\nBob: Hello. How may I help you today?\r\nUser: Please tell me the largest city in Europe.\r\nBob: Sure. The largest city in Europe is Moscow, the capital of Russia.\r\nUser:"; 

            // Load model
            var parameters = new ModelParams(modelPath)
            {
                ContextSize = 1024
            };
            using var model = LLamaWeights.LoadFromFile(parameters);

            // Initialize a chat session
            using var context = model.CreateContext(parameters);
            var ex = new InteractiveExecutor(context);
            var session = new ChatSession(ex);

            // show the prompt
            Console.WriteLine();
            Console.Write(prompt);

            // run the inference in a loop to chat with LLM
            while (true)
            {
                await foreach (var text in session.ChatAsync(prompt, 
                    new InferenceParams() { Temperature = 0.6f, AntiPrompts = new List<string> { "User:" } }))
                {
                    Console.Write(text);
                }

                Console.ForegroundColor = ConsoleColor.Green;
                prompt = Console.ReadLine();
                Console.ForegroundColor = ConsoleColor.White;
            }
        }
    }
}