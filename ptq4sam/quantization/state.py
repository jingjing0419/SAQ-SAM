import logging
from .fake_quant import QuantizeBase
logger = logging.getLogger("ptq4sam")


def enable_calibration_woquantization(model, quantizer_type='fake_quant'):
    logger.info('Enable observer and Disable quantize for {}, and disable others observer and quant'.format(quantizer_type))
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            if quantizer_type not in name:
                logger.debug('The except_quantizer is {}'.format(name))
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.debug('Enable observer and Disable quant: {}'.format(name))
            submodule.enable_observer()
            submodule.disable_fake_quant()

def enable_calibration_with_quantization(model, quantizer_type='fake_quant'):
    # 允许有其他层处于量化状态
    logger.info('Enable observer and Disable quantize for {}'.format(quantizer_type))
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            if quantizer_type not in name:
                logger.debug('The except_quantizer is {}'.format(name))
                submodule.disable_observer()
                # submodule.disable_fake_quant()
                continue
            logger.debug('Enable observer and Disable quant: {}'.format(name))
            submodule.enable_observer()
            submodule.disable_fake_quant()


def enable_quantization(model, quantizer_type='fake_quant'):
    logger.info('Disable observer and Enable quantize for {}, and disable others observer and quant'.format(quantizer_type))
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            if quantizer_type not in name:
                logger.debug('The except_quantizer is {}'.format(name))
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.info('Disable observer and Enable quant: {}'.format(name))
            submodule.disable_observer()
            submodule.enable_fake_quant()
            


def enable_quantization_plus(model, quantizer_type='fake_quant'):
    # 仅增加允许量化的层，而不关闭其他部分的量化
    logger.info('Disable observer and Enable quantize for {}'.format(quantizer_type))
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            if quantizer_type in name:
            #     logger.info('The except_quantizer is {}'.format(name))
            #     submodule.disable_observer()
            #     submodule.disable_fake_quant()
            #     continue
                logger.info('Disable observer and Enable quant: {}'.format(name))
                submodule.disable_observer()
                submodule.enable_fake_quant()

def disable_all(model):
    logger.info('Disable observer and disable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            logger.debug('Disable observer and disable quant: {}'.format(name))
            submodule.disable_observer()
            submodule.disable_fake_quant()

def disable_quantization(model):
    logger.info('Disable observer and disable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            logger.info('Disable observer and disable quantize for {}'.format(name))
            submodule.disable_observer()
            submodule.disable_fake_quant()

def disable_all_observer(model):
    logger.info('Disable observer and disable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            logger.debug('Disable observer and disable quant: {}'.format(name))
            submodule.disable_observer()
            # submodule.disable_fake_quant()
