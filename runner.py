"""
   MTTOD: runner.py

   implements train and predict function for MTTOD model.

   Copyright 2021 ETRI LIRS, Yohan Lee

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import os
import re
import copy
import math
import time
import glob
import shutil
from abc import *
from tqdm import tqdm
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from transformers.modeling_outputs import BaseModelOutput
from tensorboardX import SummaryWriter

from model import T5WithSpan, T5WithTokenSpan
from reader import MultiWOZIterator, MultiWOZReader, MultiWOZDataset, SequentialDistributedSampler, CollatorTrain
from evaluator import MultiWozEvaluator

from model import T5WithSpan, T5WithTokenSpan
from reader import MultiWOZIterator, MultiWOZReader
from evaluator import MultiWozEvaluator

from utils import definitions
from utils.io_utils import get_or_create_logger, load_json, save_json
from utils.ddp_utils import reduce_mean, reduce_sum


logger = get_or_create_logger(__name__)


class Reporter(object):
    def __init__(self, log_frequency, model_dir):
        self.log_frequency = log_frequency
        self.summary_writer = SummaryWriter(os.path.join(model_dir, "tensorboard"))

        self.global_step = 0
        self.lr = 0
        self.init_stats()

    def init_stats(self):
        self.step_time = 0.0

        self.belief_loss = 0.0
        self.span_loss = 0.0
        self.resp_loss = 0.0

        self.belief_correct = 0.0
        self.span_correct = 0.0
        self.resp_correct = 0.0

        self.belief_count = 0.0
        self.span_count = 0.0
        self.resp_count = 0.0

    def step(self, start_time, lr, step_outputs, force_info=False, is_train=True):
        self.global_step += 1
        self.step_time += (time.time() - start_time)

        self.belief_loss += step_outputs["belief"]["loss"]
        self.belief_correct += step_outputs["belief"]["correct"]
        self.belief_count += step_outputs["belief"]["count"]

        if "span" in step_outputs:
            self.span_loss += step_outputs["span"]["loss"]
            self.span_correct += step_outputs["span"]["correct"]
            self.span_count += step_outputs["span"]["count"]

            do_span_stats = True
        else:
            do_span_stats = False

        if "resp" in step_outputs:
            self.resp_loss += step_outputs["resp"]["loss"]
            self.resp_correct += step_outputs["resp"]["correct"]
            self.resp_count += step_outputs["resp"]["count"]

            do_resp_stats = True
        else:
            do_resp_stats = False

        if is_train:
            self.lr = lr
            self.summary_writer.add_scalar("lr", lr, global_step=self.global_step)

            if self.global_step % self.log_frequency == 0:
                self.info_stats("train", self.global_step, do_span_stats, do_resp_stats)

    def info_stats(self, data_type, global_step, do_span_stats=False, do_resp_stats=False):
        avg_step_time = self.step_time / self.log_frequency

        belief_ppl = math.exp(self.belief_loss / self.belief_count)
        belief_acc = (self.belief_correct / self.belief_count) * 100

        self.summary_writer.add_scalar(
            "{}/belief_loss".format(data_type), self.belief_loss, global_step=global_step)

        self.summary_writer.add_scalar(
            "{}/belief_ppl".format(data_type), belief_ppl, global_step=global_step)

        self.summary_writer.add_scalar(
            "{}/belief_acc".format(data_type), belief_acc, global_step=global_step)

        if data_type == "train":
            common_info = "step {0:d}; step-time {1:.2f}s; lr {2:.2e};".format(
                global_step, avg_step_time, self.lr)
        else:
            common_info = "[Validation]"

        belief_info = "[belief] loss {0:.2f}; ppl {1:.2f}; acc {2:.2f}".format(
            self.belief_loss, belief_ppl, belief_acc)

        if do_resp_stats:
            resp_ppl = math.exp(self.resp_loss / self.resp_count)
            resp_acc = (self.resp_correct / self.resp_count) * 100

            self.summary_writer.add_scalar(
                "{}/resp_loss".format(data_type), self.resp_loss, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/resp_ppl".format(data_type), resp_ppl, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/resp_acc".format(data_type), resp_acc, global_step=global_step)

            resp_info = "[resp] loss {0:.2f}; ppl {1:.2f}; acc {2:.2f}".format(
                self.resp_loss, resp_ppl, resp_acc)
        else:
            resp_info = ""

        if do_span_stats:
            if self.span_count == 0:
                span_acc = 0.0
            else:
                span_acc = (self.span_correct / self.span_count) * 100

            self.summary_writer.add_scalar(
                "{}/span_loss".format(data_type), self.span_loss, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/span_acc".format(data_type), span_acc, global_step=global_step)

            span_info = "[span] loss {0:.2f}; acc {1:.2f};".format(
                self.span_loss, span_acc)

        else:
            span_info = ""

        logger.info(
            " ".join([common_info, belief_info, resp_info, span_info]))

        self.init_stats()


class BaseRunner(metaclass=ABCMeta):
    def __init__(self, cfg, reader):
        self.cfg = cfg
        self.reader = reader
        self.template4resp = ["<bos_resp> sorry, could you please repeat that",\
            "<bos_resp> excuse me, could you tell me",\
                "<bos_resp> i am sorry i do not understand what you just said. please repeat the",\
                    "<bos_resp> oh, i am sorry about that. excuse me. what"]

        self.model = self.load_model()

    def load_model(self):
        if self.cfg.ckpt is not None:
            model_path = self.cfg.ckpt
            initialize_additional_decoder = False
        elif self.cfg.train_from is not None:
            model_path = self.cfg.train_from
            initialize_additional_decoder = False
        else:
            model_path = self.cfg.backbone
            initialize_additional_decoder = True

        logger.info("Load model from {}".format(model_path))

        if not self.cfg.add_auxiliary_task:
            model_wrapper = T5WithSpan
        else:
            model_wrapper = T5WithTokenSpan

        num_span = len(definitions.EXTRACTIVE_SLOT)

        model = model_wrapper.from_pretrained(model_path, num_span=num_span)

        model.resize_token_embeddings(self.reader.vocab_size)

        if initialize_additional_decoder:
            model.initialize_additional_decoder()
        
        model.to(self.cfg.device)

        if self.cfg.num_gpus > 1:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.cfg.local_rank], output_device=self.cfg.local_rank)
        
        model.to(self.cfg.device)

        return model

    def save_model(self, epoch):
        #if not os.path.exists(os.path.join(self.cfg.model_dir, "run_config.json")):
        #    save_json(self.cfg, os.path.join(self.cfg.model_dir, "run_config.json"))

        latest_ckpt = "ckpt-epoch{}".format(epoch)
        save_path = os.path.join(self.cfg.model_dir, latest_ckpt)
        
        if self.cfg.num_gpus > 1:
            model = self.model.module
        else:
            model = self.model
        
        #model = self.model

        model.save_pretrained(save_path)

        if self.cfg.save_best_model:
            self.cfg.max_to_keep_ckpt = 1

        # keep chekpoint up to maximum
        checkpoints = sorted(
            glob.glob(os.path.join(self.cfg.model_dir, "ckpt-*")),
            key=os.path.getmtime,
            reverse=True)

        checkpoints_to_be_deleted = checkpoints[self.cfg.max_to_keep_ckpt:]

        for ckpt in checkpoints_to_be_deleted:
            shutil.rmtree(ckpt)

        return latest_ckpt

    def get_optimizer_and_scheduler(self, num_traininig_steps_per_epoch, train_batch_size):
        '''
        num_train_steps = (num_train_examples *
            self.cfg.epochs) // (train_batch_size * self.cfg.grad_accum_steps)
        '''
        num_train_steps = (num_traininig_steps_per_epoch *
            self.cfg.epochs) // self.cfg.grad_accum_steps

        if self.cfg.warmup_steps >= 0:
            num_warmup_steps = self.cfg.warmup_steps
        else:
            #num_warmup_steps = int(num_train_steps * 0.2)
            num_warmup_steps = int(num_train_steps * self.cfg.warmup_ratio)

        logger.info("Total training steps = {}, warmup steps = {}".format(
            num_train_steps, num_warmup_steps))

        optimizer = AdamW(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)

        if self.cfg.no_learning_rate_decay:
            scheduler = get_constant_schedule(optimizer)
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_steps)

        return optimizer, scheduler

    def count_tokens(self, pred, label, pad_id):
        pred = pred.view(-1)
        label = label.view(-1)

        num_count = label.ne(pad_id).long().sum()
        num_correct = torch.eq(pred, label).long().sum()

        return num_correct, num_count

    def count_spans(self, pred, label):
        pred = pred.view(-1, 2)

        num_count = label.ne(-1).long().sum()
        num_correct = torch.eq(pred, label).long().sum()

        return num_correct, num_count

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError


class MultiWOZRunner(BaseRunner):
    def __init__(self, cfg):
        
        reader = MultiWOZReader(cfg.backbone, cfg.version, cfg.subversion)

        self.subversion = cfg.subversion

        self.iterator = MultiWOZIterator(reader)

        super(MultiWOZRunner, self).__init__(cfg, reader)
        self.loss_container = []

        self.uncertain_rhetorical_question_success = 0
        self.uncertain_rhetorical_question_attempt = 0
        self.uncertain_rhetorical_question_fault = 0
        self.uncertain_rhetorical_question_total = 0

        self.rhetorical_question_success = 0
        self.rhetorical_question_attempt = 0
        self.rhetorical_question_fault = 0
        self.rhetorical_question_success_tolerate = 0
        self.rhetorical_question_total = 0

    def pretty_show(self, *anything):


        def speak(something):

            if isinstance(something, str):
                self.pre_print(something)
                return

            if isinstance(something, torch.Tensor):
                something = something.tolist()
                for i in something:
                    self.pre_print(self.reader.tokenizer.decode(i))    
                    return

            for k,v in something.items():
                
                if k not in ['dial_id', 'turn_num', 'turn_domain', 'pointer', 'span', 'resp_span', 'user_span']:
                    try:
                        self.pre_print(k, self.reader.tokenizer.decode(v))
                    except:
                        self.pre_print(k, v)
                        self.pre_print("ERROR")
                        exit()
                else:
                    self.pre_print(k, v)


        for something in anything:
            speak(something)


    def step_fn(self, inputs, span_labels, belief_labels, resp_labels):
        
        def quick_show(inputs):
            for input in inputs:
                self.pre_print(self.reader.tokenizer.decode(input))

        inputs = inputs.to(self.cfg.device)
        span_labels = span_labels.to(self.cfg.device)
        belief_labels = belief_labels.to(self.cfg.device)
        resp_labels = resp_labels.to(self.cfg.device)

        attention_mask = torch.where(inputs == self.reader.pad_token_id, 0, 1)

        belief_outputs = self.model(input_ids=inputs,
                                    attention_mask=attention_mask,
                                    span_labels=span_labels,
                                    lm_labels=belief_labels,
                                    return_dict=False,
                                    add_auxiliary_task=self.cfg.add_auxiliary_task,
                                    decoder_type="belief")

        belief_loss = belief_outputs[0]
        belief_pred = belief_outputs[1]

        span_loss = belief_outputs[2]
        span_pred = belief_outputs[3]

        self.loss_container.append(belief_loss.item())
        
        if self.cfg.task == "e2e":
            last_hidden_state = belief_outputs[5]

            encoder_outputs = BaseModelOutput(last_hidden_state=last_hidden_state)

            resp_outputs = self.model(attention_mask=attention_mask,
                                      encoder_outputs=encoder_outputs,
                                      lm_labels=resp_labels,
                                      return_dict=False,
                                      decoder_type="resp")

            resp_loss = resp_outputs[0]
            resp_pred = resp_outputs[1]

            num_resp_correct, num_resp_count = self.count_tokens(
                resp_pred, resp_labels, pad_id=self.reader.pad_token_id)

        num_belief_correct, num_belief_count = self.count_tokens(
            belief_pred, belief_labels, pad_id=self.reader.pad_token_id)

        if self.cfg.add_auxiliary_task:
            num_span_correct, num_span_count = self.count_tokens(
                span_pred, span_labels, pad_id=0)

        loss = belief_loss

        if self.cfg.add_auxiliary_task and self.cfg.aux_loss_coeff > 0:
            loss += (self.cfg.aux_loss_coeff * span_loss)

        if self.cfg.task == "e2e" and self.cfg.resp_loss_coeff > 0:
            loss += (self.cfg.resp_loss_coeff * resp_loss)

        '''
        if self.cfg.num_gpus > 1:
            loss = loss.sum()
            belief_loss = belief_loss.sum()
            num_belief_correct = num_belief_correct.sum()
            num_belief_count = num_belief_count.sum()

            if self.cfg.add_auxiliary_task:
                span_loss = span_loss.sum()
                num_span_correct = num_span_correct.sum()
                num_span_count = num_span_count.sum()

            if self.cfg.task == "e2e":
                resp_loss = resp_loss.sum()
                num_resp_correct = num_resp_correct.sum()
                num_resp_count = num_resp_count.sum()
        '''

        step_outputs = {"belief": {"loss": belief_loss.item(),
                                   "correct": num_belief_correct.item(),
                                   "count": num_belief_count.item()}}

        if self.cfg.add_auxiliary_task:
            step_outputs["span"] = {"loss": span_loss.item(),
                                    "correct": num_span_correct.item(),
                                    "count": num_span_count.item()}

        if self.cfg.task == "e2e":
            step_outputs["resp"] = {"loss": resp_loss.item(),
                                    "correct": num_resp_correct.item(),
                                    "count": num_resp_count.item()}

        return loss, step_outputs

    def reduce_ddp_stepoutpus(self, step_outputs):
        step_outputs_all = {"belief": {"loss": reduce_mean(step_outputs['belief']['loss']),
                            "correct": reduce_sum(step_outputs['belief']['correct']),
                            "count": reduce_sum(step_outputs['belief']['count'])}}

        if self.cfg.add_auxiliary_task:
            step_outputs_all['span'] = {
                'loss': reduce_mean(step_outputs['span']['loss']),
                "correct": reduce_sum(step_outputs['span']['correct']),
                "count": reduce_sum(step_outputs['span']['count'])
            }

        if self.cfg.task == "e2e":
            step_outputs_all["resp"] = {
                'loss': reduce_mean(step_outputs['resp']['loss']),
                "correct": reduce_sum(step_outputs['resp']['correct']),
                "count": reduce_sum(step_outputs['resp']['count'])
            }

        return step_outputs_all

    def train_epoch(self, data_loader, optimizer, scheduler, reporter=None):
        self.model.train()
        self.model.zero_grad()
        epoch_step, train_loss = 0, 0.
        for batch in iter(data_loader):
            start_time = time.time()

            inputs, span_labels, belief_labels, resp_labels = batch

            loss, step_outputs = self.step_fn(inputs, span_labels, belief_labels, resp_labels)

            if self.cfg.grad_accum_steps > 1:
                loss = loss / self.cfg.grad_accum_steps

            loss.backward()
            train_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm)

            if (epoch_step + 1) % self.cfg.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                lr = scheduler.get_last_lr()[0]

                if reporter is not None:
                    reporter.step(start_time, lr, step_outputs)
            
            epoch_step += 1
        
        return train_loss

    def train(self):
        train_dataset = MultiWOZDataset(self.reader, 'train', self.cfg.task, self.cfg.ururu, context_size=self.cfg.context_size, num_dialogs=self.cfg.num_train_dialogs, excluded_domains=self.cfg.excluded_domains)
        
        if self.cfg.num_gpus > 1:
            train_sampler = DistributedSampler(train_dataset)
        else:
            train_sampler = sampler(train_dataset)
        
        train_collator = CollatorTrain(self.reader.pad_token_id, self.reader.tokenizer)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.cfg.batch_size_per_gpu, collate_fn=train_collator)

        num_training_steps_per_epoch = len(train_dataloader) // self.cfg.grad_accum_steps

        optimizer, scheduler = self.get_optimizer_and_scheduler(
            num_training_steps_per_epoch, self.cfg.batch_size_per_gpu)

        if self.cfg.local_rank in [0, -1]:
            reporter = Reporter(num_training_steps_per_epoch, self.cfg.model_dir)
        else:
            reporter = None

        max_score = 0.0
        for epoch in range(1, self.cfg.epochs + 1):
            train_dataloader.sampler.set_epoch(epoch)

            train_loss = self.train_epoch(train_dataloader, optimizer, scheduler, reporter)

            if self.cfg.num_gpus > 1:
                torch.distributed.barrier()
                
            logger.info("done {}/{} epoch, train loss is:{:f}".format(epoch, self.cfg.epochs, train_loss))

            # if not self.cfg.no_validation:
            #     self.validation(reporter.global_step)

            if epoch < 5: # Evaluating after training 5 epochs
                continue

            if self.cfg.save_best_model:
                current_score = self.predict()
                if max_score < current_score:
                    max_score = current_score
                    if self.cfg.local_rank in [0, -1]:
                        self.save_model(epoch)
            else:
                if self.cfg.local_rank in [0, -1]:
                    self.save_model(epoch)

            if self.cfg.num_gpus > 1:
                torch.distributed.barrier()

    def validation(self, global_step):
        self.model.eval()

        eval_dataset = MultiWOZDataset(self.reader, 'dev', self.cfg.task, self.cfg.ururu, context_size=self.cfg.context_size, excluded_domains=self.cfg.excluded_domains)
        eval_sampler = SequentialDistributedSampler(eval_dataset, self.cfg.batch_size_per_gpu_eval)
        eval_collator = CollatorTrain(self.reader.pad_token_id)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.cfg.batch_size_per_gpu_eval, collate_fn=eval_collator)

        reporter = Reporter(1000000, self.cfg.model_dir)

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Validaction"):
                start_time = time.time()

                inputs, labels = batch

                _, step_outputs = self.step_fn(inputs, *labels)

                torch.distributed.barrier()

                step_outputs_all = self.reduce_ddp_stepoutpus(step_outputs)

                if self.cfg.local_rank == 0:
                    reporter.step(start_time, lr=None, step_outputs=step_outputs_all, is_train=False)

            do_span_stats = True if "span" in step_outputs else False
            do_resp_stats = True if "resp" in step_outputs else False

            reporter.info_stats("dev", global_step, do_span_stats, do_resp_stats)

    def _train_epoch(self, train_iterator, optimizer, scheduler, reporter=None):
        self.model.train()
        self.model.zero_grad()

        for step, batch in enumerate(train_iterator):
            start_time = time.time()

            inputs, labels = batch

            _, belief_labels, _ = labels

        
            loss, step_outputs = self.step_fn(inputs, *labels)

            if self.cfg.grad_accum_steps > 1:
                loss = loss / self.cfg.grad_accum_steps

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm)

            if (step + 1) % self.cfg.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                lr = scheduler.get_last_lr()[0]

                if reporter is not None:
                    reporter.step(start_time, lr, step_outputs)

    def _train(self):
        train_batches, num_training_steps_per_epoch, _, _ = self.iterator.get_batches(
            "train", self.cfg.batch_size, self.cfg.num_gpus, shuffle=True,
            num_dialogs=self.cfg.num_train_dialogs, excluded_domains=self.cfg.excluded_domains)

        optimizer, scheduler = self.get_optimizer_and_scheduler(
            num_training_steps_per_epoch, self.cfg.batch_size)

        reporter = Reporter(self.cfg.log_frequency, self.cfg.model_dir)

        for epoch in range(1, self.cfg.epochs + 1):
            train_iterator = self.iterator.get_data_iterator(
                train_batches, self.cfg.task, self.cfg.ururu, self.cfg.add_auxiliary_task, self.cfg.context_size)

            self.train_epoch(train_iterator, optimizer, scheduler, reporter)

            logger.info("done {}/{} epoch".format(epoch, self.cfg.epochs))

            self.save_model(epoch)

            if not self.cfg.no_validation:
                self.validation(reporter.global_step)

        #self.pre_print(self.loss_container, sum(self.loss_container) / len(self.loss_container))

    def _validation(self, global_step):
        self.model.eval()

        dev_batches, num_steps, _, _ = self.iterator.get_batches(
            "dev", self.cfg.batch_size, self.cfg.num_gpus)

        dev_iterator = self.iterator.get_data_iterator(
            dev_batches, self.cfg.task, self.cfg.ururu, self.cfg.add_auxiliary_task, self.cfg.context_size)

        reporter = Reporter(1000000, self.cfg.model_dir)

        torch.set_grad_enabled(False)
        for batch in tqdm(dev_iterator, total=num_steps, desc="Validaction"):
            start_time = time.time()

            inputs, labels = batch

            _, step_outputs = self.step_fn(inputs, *labels)

            reporter.step(start_time, lr=None, step_outputs=step_outputs, is_train=False)

        do_span_stats = True if "span" in step_outputs else False
        do_resp_stats = True if "resp" in step_outputs else False

        reporter.info_stats("dev", global_step, do_span_stats, do_resp_stats)

        torch.set_grad_enabled(True)

    def finalize_bspn(self, belief_outputs, domain_history, constraint_history, span_outputs=None, input_ids=None):
        eos_token_id = self.reader.get_token_id(definitions.EOS_BELIEF_TOKEN)

        batch_decoded = []
        for i, belief_output in enumerate(belief_outputs):
            if belief_output[0] == self.reader.pad_token_id:
                belief_output = belief_output[1:]

            if eos_token_id not in belief_output:
                eos_idx = len(belief_output) - 1
            else:
                eos_idx = belief_output.index(eos_token_id)

            bspn = belief_output[:eos_idx + 1]

            decoded = {}

            decoded["bspn_gen"] = bspn

            # update bspn using span output
            if span_outputs is not None and input_ids is not None:
                span_output = span_outputs[i]
                input_id = input_ids[i]

                #self.pre_print(self.reader.tokenizer.decode(input_id))
                #self.pre_print(self.reader.tokenizer.decode(bspn))

                eos_idx = input_id.index(self.reader.eos_token_id)
                input_id = input_id[:eos_idx]

                span_result = {}

                bos_user_id = self.reader.get_token_id(definitions.BOS_USER_TOKEN)

                span_output = span_output[:eos_idx]

                b_slot = None
                for t, span_token_idx in enumerate(span_output):
                    turn_id = max(input_id[:t].count(bos_user_id) - 1, 0)
                    turn_domain = domain_history[i][turn_id]

                    if turn_domain not in definitions.INFORMABLE_SLOTS:
                        continue

                    span_token = self.reader.span_tokens[span_token_idx]

                    if span_token not in definitions.INFORMABLE_SLOTS[turn_domain]:
                        b_slot = span_token
                        continue

                    if turn_domain not in span_result:
                        span_result[turn_domain] = defaultdict(list)

                    if b_slot != span_token:
                        span_result[turn_domain][span_token] = [input_id[t]]
                    else:
                        span_result[turn_domain][span_token].append(input_id[t])

                    b_slot = span_token

                for domain, sv_dict in span_result.items():
                    for s, v_list in sv_dict.items():
                        value = v_list[-1]
                        span_result[domain][s] = self.reader.tokenizer.decode(
                            value, clean_up_tokenization_spaces=False)

                span_dict = copy.deepcopy(span_result)

                ontology = self.reader.db.extractive_ontology

                flatten_span = []
                for domain, sv_dict in span_result.items():
                    flatten_span.append("[" + domain + "]")

                    for s, v in sv_dict.items():
                        if domain in ontology and s in ontology[domain]:
                            if v not in ontology[domain][s]:
                                del span_dict[domain][s]
                                continue

                        if s == "destination" or s == "departure":
                            _s = "destination" if s == "departure" else "departure"

                            if _s in sv_dict and v == sv_dict[_s]:
                                if s in span_dict[domain]:
                                    del span_dict[domain][s]
                                if _s in span_dict[domain]:
                                    del span_dict[domain][_s]
                                continue

                        if s in ["time", "leave", "arrive"]:
                            v = v.replace(".", ":")
                            if re.match("[0-9]+:[0-9]+", v) is None:
                                del span_dict[domain][s]
                                continue
                            else:
                                span_dict[domain][s] = v

                        flatten_span.append("[value_" + s + "]")
                        flatten_span.append(v)

                    if len(span_dict[domain]) == 0:
                        del span_dict[domain]
                        flatten_span.pop()

                #self.pre_print(flatten_span)

                #input()

                decoded["span"] = flatten_span

                constraint_dict = self.reader.bspn_to_constraint_dict(
                    self.reader.tokenizer.decode(bspn, clean_up_tokenization_spaces=False))

                if self.cfg.overwrite_with_span:
                    _constraint_dict = OrderedDict()

                    for domain, slots in definitions.INFORMABLE_SLOTS.items():
                        if domain in constraint_dict or domain in span_dict:
                            _constraint_dict[domain] = OrderedDict()

                        for slot in slots:
                            if domain in constraint_dict:
                                cons_value = constraint_dict[domain].get(slot, None)
                            else:
                                cons_value = None

                            if domain in span_dict:
                                span_value = span_dict[domain].get(slot, None)
                            else:
                                span_value = None

                            if cons_value is None and span_value is None:
                                continue

                            # priority: span_value > cons_value
                            slot_value = span_value or cons_value

                            _constraint_dict[domain][slot] = slot_value
                else:
                    _constraint_dict = copy.deepcopy(constraint_dict)

                bspn_gen_with_span = self.reader.constraint_dict_to_bspn(
                    _constraint_dict)

                bspn_gen_with_span = self.reader.encode_text(
                    bspn_gen_with_span,
                    bos_token=definitions.BOS_BELIEF_TOKEN,
                    eos_token=definitions.EOS_BELIEF_TOKEN)

                decoded["bspn_gen_with_span"] = bspn_gen_with_span

            batch_decoded.append(decoded)

        return batch_decoded

    def finalize_resp(self, resp_outputs):
        bos_action_token_id = self.reader.get_token_id(definitions.BOS_ACTION_TOKEN)
        eos_action_token_id = self.reader.get_token_id(definitions.EOS_ACTION_TOKEN)

        bos_resp_token_id = self.reader.get_token_id(definitions.BOS_RESP_TOKEN)
        eos_resp_token_id = self.reader.get_token_id(definitions.EOS_RESP_TOKEN)

        batch_decoded = []
        for resp_output in resp_outputs:
            resp_output = resp_output[1:]
            if self.reader.eos_token_id in resp_output:
                eos_idx = resp_output.index(self.reader.eos_token_id)
                resp_output = resp_output[:eos_idx]

            try:
                bos_action_idx = resp_output.index(bos_action_token_id)
                eos_action_idx = resp_output.index(eos_action_token_id)
            except ValueError:
                logger.warn("bos/eos action token not in : {}".format(
                    self.reader.tokenizer.decode(resp_output)))
                aspn = [bos_action_token_id, eos_action_token_id]
            else:
                aspn = resp_output[bos_action_idx:eos_action_idx + 1]

            try:
                bos_resp_idx = resp_output.index(bos_resp_token_id)
                eos_resp_idx = resp_output.index(eos_resp_token_id)
            except ValueError:
                logger.warn("bos/eos resp token not in : {}".format(
                    self.reader.tokenizer.decode(resp_output)))
                resp = [bos_resp_token_id, eos_resp_token_id]
            else:
                resp = resp_output[bos_resp_idx:eos_resp_idx + 1]

            decoded = {"aspn_gen": aspn, "resp_gen": resp}

            batch_decoded.append(decoded)

        return batch_decoded

    def bspn2oral(self, bspn):
        return self.reader.bspn_to_constraint_dict(self.reader.tokenizer.decode(bspn, clean_up_tokenization_spaces=False))

    def oral_equal(self, orderd_dict1, orderd_dict2):
        if orderd_dict1.keys() != orderd_dict2.keys():
            return False
        
        # for k, v in orderd_dict1.keys():
            
        return True

    def is_extra_info(self, turn):
        if not isinstance(turn['user'], str):
            resp = self.reader.tokenizer.decode(turn['user'])
        
        info_template = "<bos_user> XXXX is XXXX. <eos_user>"

        import re
        
        def is_info(user, template):
            #self.pre_print(user)
            user_ = user.split()
            template_ = template.split()

            # self.pre_print(user_)
            # self.pre_print(template_)

            # self.pre_print("is_info")

            if len(user_) != len(template_):
                return False
            
            for t, tok in enumerate(template_):
                if re.sub("\.", "", tok) != "XXXX":
                    if user_[t] != tok:
                        self.pre_print("unmatch")
                        self.pre_print(tok, user_[t])
                        return False
            # self.pre_print("is!")
            return True

        return is_info(resp, info_template)

    def predict(self):
        self.pre_print(self.reader.tokenizer.get_added_vocab())
        #exit()
        self.model.eval()
        self.pre_print(self.cfg.batch_size_per_gpu_eval)

        if self.cfg.num_gpus > 1:
            model = self.model.module
        else:
            model = self.model

        pred_batches, _, _, _ = self.iterator.get_batches(
            self.cfg.pred_data_type, self.cfg.batch_size_per_gpu_eval,
            self.cfg.num_gpus, excluded_domains=self.cfg.excluded_domains)

        early_stopping = True if self.cfg.beam_size > 1 else False

        eval_dial_list = None
        if self.cfg.excluded_domains is not None:
            eval_dial_list = []

            for domains, dial_ids in self.iterator.dial_by_domain.items():
                domain_list = domains.split("-")

                if len(set(domain_list) & set(self.cfg.excluded_domains)) == 0:
                    eval_dial_list.extend(dial_ids)

        results = {}

        turn_level_acc = 0
        turn_level_total = 0

        tuple_level_acc = 0
        tuple_level_total = 0

        toy_car = 0

        for dial_batch in tqdm(pred_batches[::-1], total=len(pred_batches), desc="Prediction"):
            #self.pre_print(dial_batch[0])
            #exit(0)
            batch_size = len(dial_batch)
            toy_car += 1
            #self.pre_print("batch_size: ", batch_size, self.cfg.batch_size_per_gpu_eval)
            need_skip_tags = [0] * batch_size #

            dial_history = [[] for _ in range(batch_size)]
            domain_history = [[] for _ in range(batch_size)]
            constraint_dicts = [OrderedDict() for _ in range(batch_size)]
            for turn_batch in self.iterator.transpose_batch(dial_batch):
                #self.pre_print(len(turn_batch))
                from copy import deepcopy
                turn_batch_back = deepcopy(turn_batch)

                # for turn in turn_batch:
                    # self.pretty_show(turn)
                # self.pre_print("-----")
                batch_encoder_input_ids = []
                attention_mask = []

                for t, turn in enumerate(turn_batch):
                    # self.pretty_show(turn)
                    
                    context, _ = self.iterator.flatten_dial_history(
                        dial_history[t], [], turn["user"], self.cfg.context_size)

                    encoder_input_ids = context + turn["user"] + [self.reader.eos_token_id]

                    batch_encoder_input_ids.append(self.iterator.tensorize(encoder_input_ids))

                    turn_domain = turn["turn_domain"][-1]

                    if "[" in turn_domain:
                        turn_domain = turn_domain[1:-1]

                    domain_history[t].append(turn_domain)

                batch_encoder_input_ids = pad_sequence(batch_encoder_input_ids,
                                                       batch_first=True,
                                                       padding_value=self.reader.pad_token_id)

                batch_encoder_input_ids = batch_encoder_input_ids.to(self.cfg.device)

                attention_mask = torch.where(
                    batch_encoder_input_ids == self.reader.pad_token_id, 0, 1)

                if self.cfg.skip_when_predict:
                    for t, turn in enumerate(turn_batch):
                        if need_skip_tags[t] == 1:
                            # self.pre_print("need skip")
                            if self.is_extra_info(turn):
                                skip_attention_mask = torch.tensor([1] * len(batch_encoder_input_ids[0])).to(self.cfg.device)
                                attention_mask[t] = skip_attention_mask
                                # self.pre_print("just mask")
                                
                            else:
                                # self.pre_print("skip over!")
                                need_skip_tags[t] = 0

                # belief tracking
                with torch.no_grad():
                    self.pretty_show("__in__", batch_encoder_input_ids)
                    encoder_outputs = model(input_ids=batch_encoder_input_ids,
                                                 attention_mask=attention_mask,
                                                 return_dict=False,
                                                 encoder_only=True,
                                                 add_auxiliary_task=self.cfg.add_auxiliary_task)

                    span_outputs, encoder_hidden_states = encoder_outputs

                    if isinstance(encoder_hidden_states, tuple):
                        last_hidden_state = encoder_hidden_states[0]
                    else:
                        last_hidden_state = encoder_hidden_states

                    # wrap up encoder outputs
                    encoder_outputs = BaseModelOutput(
                        last_hidden_state=last_hidden_state)
                    
                    belief_outputs = model.generate(encoder_outputs=encoder_outputs,
                                                         attention_mask=attention_mask,
                                                         eos_token_id=self.reader.eos_token_id,
                                                         max_length=200,
                                                         do_sample=self.cfg.do_sample,
                                                         num_beams=self.cfg.beam_size,
                                                         early_stopping=early_stopping,
                                                         temperature=self.cfg.temperature,
                                                         top_k=self.cfg.top_k,
                                                         top_p=self.cfg.top_p,
                                                         decoder_type="belief")

                belief_outputs = belief_outputs.cpu().numpy().tolist()

                if self.cfg.add_auxiliary_task:
                    pred_spans = span_outputs[1].cpu().numpy().tolist()

                    input_ids = batch_encoder_input_ids.cpu().numpy().tolist()
                else:
                    pred_spans = None
                    input_ids = None
                
                # for outputs in belief_outputs:
                    # self.pre_print("***", self.reader.tokenizer.decode(outputs))
                
                decoded_belief_outputs = self.finalize_bspn(
                    belief_outputs, domain_history, constraint_dicts, pred_spans, input_ids)

                for t, turn in enumerate(turn_batch):
                    turn.update(**decoded_belief_outputs[t])

                if self.cfg.task == "e2e":
                    dbpn = []

                    if self.cfg.use_true_dbpn:
                        for turn in turn_batch:
                            dbpn.append(turn["dbpn"])
                    else:
                        for turn in turn_batch:
                            if self.cfg.add_auxiliary_task:
                                bspn_gen = turn["bspn_gen_with_span"]
                            else:
                                bspn_gen = turn["bspn_gen"]

                            bspn_gen = self.reader.tokenizer.decode(
                                bspn_gen, clean_up_tokenization_spaces=False)
                            #self.pre_print("**", bspn_gen)
                            db_token = self.reader.bspn_to_db_pointer(bspn_gen,
                                                                      turn["turn_domain"])

                            dbpn_gen = self.reader.encode_text(
                                db_token,
                                bos_token=definitions.BOS_DB_TOKEN,
                                eos_token=definitions.EOS_DB_TOKEN)

                            turn["dbpn_gen"] = dbpn_gen

                            dbpn.append(dbpn_gen)

                    for t, db in enumerate(dbpn):
                        if self.cfg.use_true_curr_aspn:
                            db += turn_batch[t]["aspn"]

                        # T5 use pad_token as start_decoder_token_id
                        dbpn[t] = [self.reader.pad_token_id] + db

                    # aspn has different length
                    if self.cfg.use_true_curr_aspn:
                        for t, _dbpn in enumerate(dbpn):
                            resp_decoder_input_ids = self.iterator.tensorize([_dbpn])

                            resp_decoder_input_ids = resp_decoder_input_ids.to(self.cfg.device)

                            encoder_outputs = BaseModelOutput(
                                last_hidden_state=last_hidden_state[t].unsqueeze(0))

                            with torch.no_grad():
                                self.pretty_show("__in__", resp_decoder_input_ids)
                                resp_outputs = model.generate(
                                    encoder_outputs=encoder_outputs,
                                    attention_mask=attention_mask[t].unsqueeze(0),
                                    decoder_input_ids=resp_decoder_input_ids,
                                    eos_token_id=self.reader.eos_token_id,
                                    max_length=300,
                                    do_sample=self.cfg.do_sample,
                                    num_beams=self.cfg.beam_size,
                                    early_stopping=early_stopping,
                                    temperature=self.cfg.temperature,
                                    top_k=self.cfg.top_k,
                                    top_p=self.cfg.top_p,
                                    decoder_type="resp")

                                resp_outputs = resp_outputs.cpu().numpy().tolist()

                                decoded_resp_outputs = self.finalize_resp(resp_outputs)

                                turn_batch[t].update(**decoded_resp_outputs[0])
                    else:
                        resp_decoder_input_ids = self.iterator.tensorize(dbpn)

                        resp_decoder_input_ids = resp_decoder_input_ids.to(self.cfg.device)

                        # response generation
                        with torch.no_grad():
                            self.pretty_show("__in__, no true",resp_decoder_input_ids)
                            resp_outputs = model.generate(
                                encoder_outputs=encoder_outputs,
                                attention_mask=attention_mask,
                                decoder_input_ids=resp_decoder_input_ids,
                                eos_token_id=self.reader.eos_token_id,
                                max_length=300,
                                do_sample=self.cfg.do_sample,
                                num_beams=self.cfg.beam_size,
                                early_stopping=early_stopping,
                                temperature=self.cfg.temperature,
                                top_k=self.cfg.top_k,
                                top_p=self.cfg.top_p,
                                decoder_type="resp")

                        resp_outputs = resp_outputs.cpu().numpy().tolist()

                        decoded_resp_outputs = self.finalize_resp(resp_outputs)

                        for turn in turn_batch:
                            self.pretty_show(turn)
                            self.pre_print("before")

                        for t, turn in enumerate(turn_batch):
                            turn.update(**decoded_resp_outputs[t])
                            self.check_update(turn_batch_back[t], turn) 
                            #need_skip = self.catch(turn)
                            need_skip = self.catch_bspn(turn["bspn"], turn["bspn_gen"])
                            self.catch(turn)
                            #if need_skip:
                            #    need_skip_tags[t] = 1


                        for turn in turn_batch:
                             self.pretty_show(turn)
                             self.pre_print("after")

                        self.pre_print('$'*10)
                        
                # update dial_history
                for t, turn in enumerate(turn_batch):
                    pv_text = copy.copy(turn["user"])

                    if self.cfg.use_true_prev_bspn:
                        pv_bspn = turn["bspn"]
                    else:
                        if self.cfg.add_auxiliary_task:
                            pv_bspn = turn["bspn_gen_with_span"]
                        else:
                            pv_bspn = turn["bspn_gen"]

                    if self.cfg.use_true_dbpn:
                        pv_dbpn = turn["dbpn"]
                    else:
                        pv_dbpn = turn["dbpn_gen"]

                    if self.cfg.use_true_prev_aspn:
                        pv_aspn = turn["aspn"]
                    else:
                        pv_aspn = turn["aspn_gen"]

                    if self.cfg.use_true_prev_resp:
                        if self.cfg.task == "e2e":
                            pv_resp = turn["redx"]
                        else:
                            pv_resp = turn["resp"]
                    else:
                        pv_resp = turn["resp_gen"]

                    if self.cfg.ururu:
                        pv_text += pv_resp
                    else:
                        pv_text += (pv_bspn + pv_dbpn + pv_aspn + pv_resp)

                    dial_history[t].append(pv_text)

            #if toy_car == 10:
            #    exit()

            result = self.iterator.get_readable_batch(dial_batch)
            results.update(**result)

        #self.pre_print("turn_level_acc: ", turn_level_acc / turn_level_total)

        # self.pre_print("%"*10)

        self.turn_level_uncertain_active_question_precison = self.uncertain_rhetorical_question_success / (self.uncertain_rhetorical_question_attempt+1e-10)
        self.turn_level_uncertain_active_question_recall = self.uncertain_rhetorical_question_success / (self.uncertain_rhetorical_question_total + 1e-10)
        self.turn_level_uncertain_active_question_f1_score = 2 * self.turn_level_uncertain_active_question_precison * self.turn_level_uncertain_active_question_recall / \
            (self.turn_level_uncertain_active_question_precison + self.turn_level_uncertain_active_question_recall + 1e-10) 

        self.pre_print("attempt (uncertain): ", self.uncertain_rhetorical_question_attempt)
        self.pre_print("fault (uncertain): ", self.uncertain_rhetorical_question_fault)
        self.pre_print("success (uncertain): ", self.uncertain_rhetorical_question_success)
        self.pre_print("total (uncertain): ", self.uncertain_rhetorical_question_total)

        self.pre_print("turn_level_uncertain_active_question_acc (uncertain): ", self.turn_level_uncertain_active_question_precison )
        self.pre_print("turn_level_uncertain_active_question_recall (uncertain): ", self.turn_level_uncertain_active_question_recall )
        self.pre_print("turn_level_uncertain_active_question_f1_score (uncertain): ", self.turn_level_uncertain_active_question_f1_score)

        self.pre_print("%"*10)

        self.turn_level_active_question_precison_tolerate = self.rhetorical_question_success_tolerate / (self.rhetorical_question_attempt+1e-10)
        self.turn_level_active_question_recall_tolerate = self.rhetorical_question_success_tolerate / (self.rhetorical_question_total + 1e-10)
        self.turn_level_active_question_f1_score_tolerate = 2 * self.turn_level_active_question_precison_tolerate * self.turn_level_active_question_recall_tolerate / \
            (self.turn_level_active_question_precison_tolerate + self.turn_level_active_question_recall_tolerate + 1e-10)
        
        self.pre_print("attempt (template): ", self.rhetorical_question_attempt)
        self.pre_print("fault (template): ", self.rhetorical_question_fault)
        self.pre_print("success (template): ", self.rhetorical_question_success)
        self.pre_print("success_tolerate (template): ", self.rhetorical_question_success_tolerate)
        self.pre_print("total (template): ", self.rhetorical_question_total)

        self.pre_print("turn_level_active_question_acc_tolerate (template): ", self.turn_level_active_question_precison_tolerate )
        self.pre_print("turn_level_active_question_recall_tolerate (template): ", self.turn_level_active_question_recall_tolerate)
        self.pre_print("turn_level_active_question_f1_score_tolerate (template): ", self.turn_level_active_question_f1_score_tolerate)

        self.turn_level_active_question_precison = self.rhetorical_question_success / (self.rhetorical_question_attempt+1e-10)
        self.turn_level_active_question_recall = self.rhetorical_question_success / (self.rhetorical_question_total + 1e-10)
        self.turn_level_active_question_f1_score = 2 * self.turn_level_active_question_precison * self.turn_level_active_question_recall / \
            (self.turn_level_active_question_precison + self.turn_level_active_question_recall + 1e-10)

        self.pre_print("turn_level_active_question_acc: ", self.turn_level_active_question_precison )
        self.pre_print("turn_level_active_question_recall: ", self.turn_level_active_question_recall )
        self.pre_print("turn_level_active_question_f1_score: ", self.turn_level_active_question_f1_score)

        # self.pre_print("%"*10)

        if self.cfg.output:
            save_json(results, os.path.join(self.cfg.ckpt, self.cfg.output))

        evaluator = MultiWozEvaluator(self.reader, self.cfg.pred_data_type, self.subversion)

        if self.cfg.task == "e2e":
            bleu, success, match = evaluator.e2e_eval(
                results, eval_dial_list=eval_dial_list, add_auxiliary_task=self.cfg.add_auxiliary_task)

            score = 0.5 * (success + match) + bleu

            logger.info('match: %2.2f; success: %2.2f; bleu: %2.2f; score: %.2f' % (
                match, success, bleu, score))
        else:
            joint_goal, f1, accuracy, count_dict, correct_dict = evaluator.dialog_state_tracking_eval(
                results, add_auxiliary_task=self.cfg.add_auxiliary_task)

            logger.info('joint acc: %2.2f; acc: %2.2f; f1: %2.2f;' % (
                joint_goal, accuracy, f1))

            for domain_slot, count in count_dict.items():
                correct = correct_dict.get(domain_slot, 0)

                acc = (correct / count) * 100

                logger.info('{0} acc: {1:.2f}'.format(domain_slot, acc))
    
        return score

    def check_update(self, turn_back, turn):
        for k,v in turn_back.items():
            if v != turn[k]:
                self.pre_print('%'*10)
                self.pre_print("update:")
                self.pre_print(self.reader.tokenizer.decode(v))
                self.pre_print(self.reader.tokenizer.decode(turn[k]))
                self.pre_print('%'*10)

    def get_key_value(self, template, resp):
        # matched already
        import re
        import string
        re_match = re.match(template, resp)
        # <_sre.SRE_Match object; span=(0, 3), match='sss'>
        cut = resp[re_match.span()[-1]:] # "sssss XXXXX sdadsadda"
        key_value = re.sub('[{}]'.format(string.punctuation), "", cut.split()[0])
        # self.pre_print(key_value) 
        return key_value

    def extract(self, bspn):
        tmp = bspn.split()
        return  "true" == tmp[3] if len(tmp) > 4 else False

    def catch_bspn(self, bspn, bspn_gen):
        """
        """
        self.pre_print("***** uncertain catcher *****")
        if not isinstance(bspn, str):
            bspn = self.reader.tokenizer.decode(bspn)
        if not isinstance(bspn_gen, str):
            bspn_gen = self.reader.tokenizer.decode(bspn_gen)
        
        #uncertainty_token_true = "<bos_belief> [uncertain] [value_bool] true"
        #uncertainty_token_false = "<bos_belief> [uncertain] [value_bool] false"

        self.pre_print("bspn", bspn)
        self.pre_print("bspn_gen", bspn_gen)

        if self.extract(bspn):#uncertainty_token_true in bspn:
            self.uncertain_rhetorical_question_total += 1
            self.pre_print("Need Catch")
            if self.extract(bspn_gen):#uncertainty_token_true in bspn_gen:
                self.uncertain_rhetorical_question_attempt += 1
                self.uncertain_rhetorical_question_success += 1
                self.pre_print("Catch!")
                return False
            else:
                self.pre_print("Not catch")
                return True

        else:
            if self.extract(bspn_gen):#uncertainty_token_true in bspn_gen:
                self.uncertain_rhetorical_question_attempt += 1
                self.uncertain_rhetorical_question_fault += 1
                self.pre_print("Fault")
                return False

    def catch(self, turn):

        """
        """
        
        resp = turn["resp"]
        resp_gen = turn["resp_gen"]
        
        self.pre_print("***** template catcher *****")
        if not isinstance(resp, str):
            resp = self.reader.tokenizer.decode(resp)
        if not isinstance(resp_gen, str):
            resp_gen = self.reader.tokenizer.decode(resp_gen)


        self.pre_print("resp", resp)
        self.pre_print("resp_gen", resp_gen)

        import re
        for template in self.template4resp:
            if re.match(template, resp):
                self.rhetorical_question_total += 1
                self.pre_print("Need Catch")
                self.pre_print(self.reader.tokenizer.decode(turn["user"]))

                for template2 in self.template4resp:
                    if re.match(template2, resp_gen):
                        self.rhetorical_question_attempt += 1
                        self.rhetorical_question_success_tolerate += 1
                        # return False
                        if self.get_key_value(template, resp) == self.get_key_value(template2, resp_gen):
                            self.rhetorical_question_success += 1
                            self.pre_print("Catch!")
                            return False

                else:
                    self.pre_print("Not catch")
                    self.pre_print(resp_gen)
                    return True

        for template in self.template4resp:
            if re.match(template, resp):
                self.rhetorical_question_attempt += 1
                self.rhetorical_question_fault += 1
                
                self.pre_print("Fault")
                return False

    def pre_print(self, *value):
        if int(self.subversion) >= 0:
            for v in value:
                print(v)
        
