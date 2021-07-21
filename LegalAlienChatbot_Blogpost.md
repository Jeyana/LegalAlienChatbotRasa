# LegalAlienChatbot: your AI language buddy, powered by Rasa Open Source

**Techies**: Jeyana Morozenko, Violetta Shishkina, Nazlı Dolu *(team TalkingHeads)*

**Mentor**: Matthijs Rijm

![](images/telegram_new.jpg)

The first three sections of this blog post explain the motivation for creating **LegalAlienChatbot** and provide some Machine Learning background. [Our Experience with Rasa Open Source](#our-experience-with-rasa-open-source) section is an informally written deep dive into our journey and the technical aspects of building a chatbot on a popular free platform.

## Abstract

With our project, we aim to build a chatbot to help people overcome foreign language anxiety, which is a big part of a language barrier. This is mainly the feeling of unease, worry, nervousness and apprehension experienced in learning or using a foreign language. We believe that before directly talking with a person, some written exercise with a human-like chatbot could ease the foreign language anxiety and increase the pace of language learning. For this purpose, we used a natural language understanding model. We created a chatbot using Rasa, an open source conversational AI platform.

Unlike Grammarly, LegalAlien does not correct user's messages, but it is pretty good at understanding bad English. It can answer questions about language learning, help you develop a study plan, or simply chitchat with you. LegalAlien is available 24/7, it is never annoyed by the mistakes you make, and you can never accidentally offend it.

## Introduction

It is quite natural for people to be afraid of starting to speak a foreign language, even if they already understand a lot. This may have a significant effect on their confidence and self-esteem causing individuals to be quieter and less willing to communicate.

Language barrier is a broad term for this kind of a roadblock and it hinders communication tremendously. There are several underlying reasons, some of them are linked to experience (or language proficiency) and others are purely psychological. The fear of not being understood, the fear of misunderstanding the interlocutor, the “shame” of the non-native accent can be a huge barrier on the way to speaking a foreign language and enjoying it. If on top of that, we have a person who is an introvert, or has low self-esteem, this barrier can become an impenetrable wall that might make the person give up the idea of speaking a new language.  

One piece of advice that people with foreign language anxiety are given is to imagine how the discussion might go and think about what they want to say. Great! Prepare the required phrases and play out the scenario as if you were actually using them. Then we thought, why not do this with the help of technology?

Our goal was to build a chatbot that is fun to talk to. We thought that if the chatbot could imitate a person, users could improve themselves in personal communication until they felt confident enough to talk to real people. To achieve this, we worked with Rasa which is an open source machine learning framework for automated text and voice-based conversations. It understands user messages and constructs conversations.

So, if you are a learner of English as a foreign language, who knows basic grammar and has enough vocabulary to build simple sentences, desperately needs practice but does not feel confident to communicate with other English speakers, reach out to us and try LegalAlien! 

Our chatbot is a language buddy that is always there for you. It can help you transition from theory to practice in a fun way and on your own schedule. Learning should not be boring, otherwise it brings no long-term results. The fear of using the language can easily make any learner lose their motivation and give up. 

Remember, you are not alone on this journey of learning something new! 

## Method

In order to create a chatbot before the development of machine learning algorithms, one had to write a rule based program, where all the user messages and bot responses are hardcoded. This would require the programmer to think of every possible way that a user might say something so that it could map the correct response of the chatbot. There are two downsides of this method. One of them is the fact that it is not possible to think of every possible sentence a user might write. Another one is, even though we assume that the programmer “somehow” finds out the way of enumerating all the sentences, the user might make a typo and the algorithm won’t be able to understand this. To avoid these downsides and actually achieve what we expect our chatbot to do, we examined natural language programming (NLP) models. 

NLP is a subfield of artificial intelligence (AI) which deals with methods of programming computers to process large amounts of natural language data. However, the chatbot needs to understand what the user is saying, and it is also required to decide what to say in response. Hence, it should understand the context of data (what the user means when they are saying something). To achieve this, natural language understanding (NLU) takes the center stage. NLU is a subfield of NLP. While NLP ensures the transformation of large amounts of unstructured natural language texts into structured data, NLU focuses on deriving insights from this structured data. The use of NLU allows chatbots to comprehend the context.

To utilize an NLU model, we used Rasa which is an open source machine learning platform.

The Rasa process works as follows: 

1. The user writes a message by using a messaging channel such as Telegram, Facebook Messenger, Slack etc.

2. This message is given as an input to the NLU model which transforms the message into a machine readable format. The NLU model makes a prediction to classify the meaning of the message (intent) and returns this output to the Dialogue Management model.

3. The Dialogue Management Model decides what the chatbot should do next (action).

4. Then Rasa Open Source chooses a response to give back to the user based on the set of responses.

Rasa is able to utilize different machine learning policies to decide which action to take. The ones we implemented are Rule Policy, The Transformer Embedding Dialogue (TED) Policy and Memoization Policy. Rule Policy is used for relatively simple cases in which the model needs to classify only one message of the user. On the other hand, TED and Memoization Policies are utilized for more complex cases. The parameters of these policies can be adjusted manually through the config.yml file. For example, we experimented with the number of epochs for DIETClassifier and ResponseSelector.

## Our Experience with Rasa Open Source






**Training the NLU Model**

NLU training data is formed of intents which are basically the categorization of the possible user messages, and entities. Intents consist of training examples that represent all of the different ways a user might express the intent. Here is an example of an intent and training examples. 

![](images/ask-how-to-improve-eng.png)

Here, the intent category “ask_how_to_improve_english” is generated to define possible different ways of how a user might ask “How to improve my English?”. All of the possible example questions are mainly the training examples. One significant advantage of Rasa is that since the policies we use are pretrained, we do not need to provide hundreds of example questions but five of them is sufficient for the training of the NLU model. Intents are defined in the nlu.md file.

These are other examples of intents we have generated. These are used more than once in the algorithms.

![](images/general-intent-example.png)

These intents, however, are only used for specific stories, in this case FAQ2.

![](images/nlu-intent-example.png)

Also, Rasa entities and slots allow us to extract specific information from user messages. In nlu.yml, we specify how to extract entities. Check out the following example, the entity in this case is called "name".

![](images/entity.png)

Later in domain.yml, we define entities and slots, and write bot responses using slots. As you see, the entity and the slot have the same name. It makes it easier to fill the slot automatically. Rasa automatically stores entities into the slots of the same name.

![](images/entity-slot.png)

We also need to specify possible responses that a chatbot can say to a user. These responses are hardcoded and can be found in the domain.yml file. As you can see, we created a response to the previously introduced intent “ask_how_to_improve_english” and this response is defined as “utter_ways_to_improve_english”.

![](images/response.png)

If we continue on the same example of a user asking how to improve her English, we now have both the intent which helps chatbot to understand what the user is meaning to ask and its response to this question. It is time to show the chatbot that which intents and responses are correlated. To do so, we may create rules.

![](images/rule-example.png)

Hence, if a user asks a question about how to improve English, our chatbot will understand this intent and take the following action which is to say the sentence defined in “utter_ways_to_improve_english”. We have also created other responses for the “ask_how_to_improve_english” intent.

![](images/responses.png)

**Training the Dialogue Management Model**

We also need to train the dialogue management model. For this, the model requires rule or story data which are the combinations of both the intent behind what the user says and the chatbot response. This training is mainly to teach the model what to say/do depending on what the user said so far.

Then we come up with stories for the chatbot to construct dialogue like conversations by using TED Policy. The stories are created in stories.md file. An example of how to create stories is as follows.

![](images/stories-FAQ2.png)

Different paths are defined for a conversation which starts by the user asking how much practice she should have. In all of the stories, at first the chatbot responds by asking whether one hour in five days would be realistic for the user (using the action: utter_is_one_hour_five_days_realistic). However the response of the user to this action may differ in real life, and stories tried to capture these various patterns. In the first story, the user chooses to go with what the chatbot suggests and hence the chatbot encourages the user in responses. In the second story, apparently what the chatbot suggests is too much for the user, and so it asks for an alternative plan. In the third story, the user says something irrelevant, and the chatbot is not insisting on answering the practice plan question but rather suggesting to talk about something else.

For all these input data for the training model, we used Grammarly to correct our English for the chatbot’s responses as none of us is a native English speaker.

In the end we had the following conversation with our chatbot:

![](images/your-input.png)

As you can see, even though the sentences of the user are not grammatically correct, the chatbot correctly identifies the intent and gives a proper response.

### Advantages of Rasa

- The computational time of training is faster than simple benchmark chatbots.

- Even when the user message is grammatically incorrect or contains typos, the model is able to understand the intent of the user.

- Having two separate models (NLU and Dialogue Management Model) saves time. If you make a change that has to do with only one of these models, you don't need to retrain the other one.

- The fact that it is an open-source project gives the programmer high flexibility. If you don't like something about Rasa, you are very welcome to improve it.

- Rasa is entirely written in Python! (also integrated with Python 3.8)

- Training data is written in YAML files which are human-readable.

### Disadvantages of Rasa

- Debugging and collaboration can be a nightmare. Sometimes the errors are not caught, or the error messages are irrelevant and unhelpful, and you have to guess what actually went wrong.
- The syntax for YAML files changes from version to version with no backward compatibility.
- The documentation is incomplete. The "Rasa for Beginners" course on Udemy is outdated. A lot of "figuring out" must happen before you get anything to work.
- Machine Learning happens only when Rasa recognizes user messages and predicts bot actions. The actual bot messages are hardcoded, not generated (although they can contain variables extracted from user messages or generated with Python code).

## Conclusion

Rasa might be one of the best **free** conversational AI platforms out there, and we had a lot of fun creating this project together. Still, we wouldn't recommend Rasa for teams with more than one programmer, or for large and complex chatbots.

With this project, we also wanted to show that AI is an awesome tool that can help people acquire new skills. We even asked a GPT3-based AI what it thought about a bot that teaches English. Here’s the response:

“The idea is not to replace human teachers, but to provide some extra help to those who need it”. 

And we absolutely agree. 

## Chat with LegalAlien!

You can search for **@LegalAlienChatbot** on Telegram and write a message.

If you run into problems or want to share your experience, send an email to **legalalienchatbot@gmail.com**

Thank you for your interest!

