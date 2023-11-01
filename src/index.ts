/* eslint-disable linebreak-style */
/* eslint-disable quotes */
/* eslint-disable new-cap */
/* eslint-disable max-len */
import * as express from "express";
import * as cors from "cors";
import {ChatOpenAI} from "langchain/chat_models/openai";
// import {ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate} from "langchain/prompts";
// import * as functionsV2 from "firebase-functions/v2/https";
// import {HumanMessage} from "langchain/schema";
import {OpenAIEmbeddings} from "langchain/embeddings/openai";
import * as functions from "firebase-functions";
import {Pinecone} from "@pinecone-database/pinecone";
import {Document} from "langchain/document";
import {PineconeStore} from "langchain/vectorstores/pinecone";
import {loadQAStuffChain} from "langchain/chains";

const app = express();
app.use(cors({origin: true}));
app.use(express.json());
const pinecone = new Pinecone({
  apiKey: "311493b6-eb58-46ca-a419-b2dfe3e22fad",
  environment: "gcp-starter",
});
const pineconeIndex = pinecone.Index("diarify");

exports.vectorDiaryEntry = functions.https.onCall(async (data) => {
  if (!data.entry || !data.user) {
    throw new functions.https.HttpsError("invalid-argument", "an entry is required ...");
  }
  const dateNow = new Date().toLocaleString();
  const entry = `Date and Time:\n${dateNow}\n
  Entry:\n${data.entry}`;
  const documents = [
    new Document({
      metadata: {user: data.user},
      pageContent: entry,
    }),
  ];
  const embeddings = new OpenAIEmbeddings({openAIApiKey: process.env.OPENAI});
  const docSearch = await PineconeStore.fromDocuments(documents, embeddings, {
    pineconeIndex,
  });
  return {"response": "Sucessfully added to the vector db!", "data": docSearch};
});

exports.chatWithDiary = functions.https.onCall(async (data) => {
  if (!data.question || !data.user) {
    throw new functions.https.HttpsError("invalid-argument", "an question is required ...");
  }

  const vectorStore = await PineconeStore.fromExistingIndex(
    new OpenAIEmbeddings({openAIApiKey: process.env.OPENAI}),
    {pineconeIndex}
  );
  const dateNow = new Date().toLocaleString();
  const question = `Date and Time:\n${dateNow}\n
  Question:\n${data.question}`;
  const vectorEntryResponse = await vectorStore.similaritySearch(question, 10, {user: data.user});
  const llm = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    openAIApiKey: process.env.OPENAI,
    temperature: 0.9,
  });
  const stuffChain = loadQAStuffChain(llm);
  const stuffResult = await stuffChain.call({
    input_documents: vectorEntryResponse,
    question: question,
  });
  return {"response": stuffResult};
});
