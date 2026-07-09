实验结果基于 bq_corpus 数据集

## 基本的实验结论：
### BERT方法：
对于文本匹配的场景，总体上CrossEncoder要比BiEncoder准确率高，但是CrossEncoder推理速度稍慢，适用于少量数据精排；
BiEncoder推理速度快，但是精度稍低，适用于大数据量粗召回。
BiEncoder中CosineEmbeddingLoss训练方法在小训练集上训练出来的模型精度更高，TripletLoss训练方法在大数据集上训练出来的模型精度更高。

### LLM方法：
LLM的方法，无论是prompt还是微调，总体上精度都要比BERT方法低。
其中prompt方法最低，尤其是F1分数比微调低很多，当然通过prompt优化也许会有提升。
微调方法中Lora微调比全量微调低3-5个百分点。

### 总结：
所以在文本匹配场景中，LLM在准确率上不占优势，没有BERT模型效果好。
LLM更多用于冷启动，对数据做预标注，结合置信度进行人工review，提升标注效率，产生高质量标注数据，用于小模型训练。
