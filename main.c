/**
 * \file
 *
 * \brief Empty user application template
 *
 */

/**
 * \mainpage User Application template doxygen documentation
 *
 * \par Empty user application template
 *
 * Bare minimum empty user application template
 *
 * \par Content
 *
 * -# Include the ASF header files (through asf.h)
 * -# "Insert system clock initialization code here" comment
 * -# Minimal main function that starts with a call to board_init()
 * -# "Insert application code here" comment
 *
 */

/*
 * Include header files for all drivers that have been imported from
 * Atmel Software Framework (ASF).
 */
/*
 * Support and FAQ: visit <a href="https://www.microchip.com/support/">Microchip Support</a>
 */
#include <stdio.h>
#include <asf.h>

// PORT C
#define PWM1 (1<<0)
#define PWM2 (1<<1)

#define QDEC1 (1<<2)
#define QDEC2 (1<<3)

void initQdec(qdec_config_t* config);
void initUart(void);
void sendMsg(char* data);
void initLed(void);
uint8_t getChar(void);
void putChar(uint8_t chr);
void readPosition(char* data);

//typedef struct qdec_config {
//	PORT_t *port; // PORT to use 
//	uint8_t pins_base; // Base pin for phases (A/B) connected
//	uint16_t pins_filter_us;  // Digital filter time (2 µs filtering)
//	struct {
//		bool pins_invert; // No inversion for phase pins
//	} phases;
//	struct {
//		bool enabled;  // Enable index signal
//		bool pin_invert;
//		enum QDec_index_rec_state rec_state;
//	} index;
//	bool rotary;  // Enable rotary decoding
//	uint8_t event_channel;  // Event system channel 0
//	volatile void *timer;  // Timer used for counting quadrature signals
//	uint16_t revolution;  // Number of encoder revolutions for calibration
//	struct {
//		bool enabled; // Disable frequency measurement
//		uint8_t event_channel; // Event channel for frequency measurement (Channel 2)
//		volatile void *timer; // Timer for frequency measurement (TCC1)
//		uint32_t unit; // Frequency unit for calculation
//		uint32_t coef;
//		uint16_t last_freq;
//	} freq_opt;
//} qdec_config_t;

static qdec_config_t config_proximal = 
{
	.port = &PORTA,
	.pins_base = 0,
	.pins_filter_us = 2,
	.phases.pins_invert = false,
	.index.enabled = true,
	.rotary = true,
	.event_channel = 0,
	.timer = &TCC0,
	.freq_opt.timer = &TCC1,
	.freq_opt.enabled = false,
	.freq_opt.event_channel = 2,
	.freq_opt.unit = 1000,
	.revolution = 4
};
static qdec_config_t config_medial =
{
	.port = &PORTA,
	.pins_base = 0,
	.pins_filter_us = 2,
	.phases.pins_invert = false,
	.index.enabled = true,
	.rotary = true,
	.event_channel = 0,
	.timer = &TCC0,
	.freq_opt.timer = &TCC1,
	.freq_opt.enabled = false,
	.freq_opt.event_channel = 2,
	.freq_opt.unit = 1000,
	.revolution = 4
};
static qdec_config_t config_distal =
{
	.port = &PORTC,
	.pins_base = 5,
	.pins_filter_us = 2,
	.phases.pins_invert = false,
	.index.enabled = true,
	.rotary = true,
	.event_channel = 0,
	.timer = &TCC1,
	.freq_opt.timer = &TCC1,
	.freq_opt.enabled = false,
	.freq_opt.event_channel = 2,
	.freq_opt.unit = 1000,
	.revolution = 4
};

ISR(USARTE0_RXC_vect)
{
	PORTA.OUT |= (1 << 7);
	uint8_t data = USARTE0.DATA;
	if (data == 'r')
	{
//		uint16_t qdec_position_prox = qdec_get_position(&config_proximal) / 2;
//		bool dir_prox = qdec_get_direction(&config_proximal);
//		uint16_t qdec_position_med = qdec_get_position(&config_medial) / 2;
//		bool dir_med = qdec_get_direction(&config_medial);
		uint16_t qdec_position_dist = qdec_get_position(&config_distal) / 2;
		bool dir_dist = qdec_get_direction(&config_distal);
		char msg[200];
		sprintf(msg, "%d, %d\n", qdec_position_dist, dir_dist);
//		sprintf(msg, "%d, %d, %d, %d, %d, %d\n", qdec_position_prox, dir_prox, qdec_position_med, dir_med, qdec_position_dist, dir_dist);
		sendMsg(msg);
	}
	else
	{
		sendMsg("online\n");
	}
	//printf(" %5umHz\r\n", qdec_get_frequency(&config_proximal));
	delay_ms(250);
	PORTA.OUT &= (0 << 7);
}

// int main (void)
// {
// 	initLed();
// 	initUart();
// //	initQdec(&config_proximal);
// //	initQdec(&config_medial);
// 	initQdec(&config_distal);
// 	
// 	
// 	char msg[200];
// 	sprintf(msg, "%s\n", "Ready");
// 	sendMsg(msg);
// 	delay_ms(250);
// 			
// 	while (1)
// 	{
// 		char data[200];
// 		readPosition(data);
// 		sendMsg(data);
// 		delay_ms(100);
// 	}
// }

void readPosition(char* data)
{
	uint16_t qdec_position = qdec_get_position(&config_distal) / 2;
	bool dir = qdec_get_direction(&config_distal);
	sprintf(data, "%d, %d\n", qdec_position, dir);
}

void initQdec(qdec_config_t* config)
{
	// configure PORTC5 and PORTC6 to detect level changes
	PORTC_PIN5CTRL=PORT_ISC_LEVEL_gc; // AMT103 A
	PORTC_PIN6CTRL=PORT_ISC_LEVEL_gc; // AMT103 B 
	//  configures PORTC7 to detect both rising and falling edges for the index signal
	PORTC_PIN7CTRL=PORT_ISC_BOTHEDGES_gc; // AMT103 X
	// timer period
	TCC1_PER=19999;
	// Event System Channel Mux Configuration
	EVSYS_CH0MUX=EVSYS_CHMUX_PORTC_PIN5_gc; // Connect A to Event Channel 0
	EVSYS_CH1MUX=EVSYS_CHMUX_PORTC_PIN7_gc; // Connect B to Event Channel 1
	// enables the quadrature decoder for channel 0 and applies digital filtering
	EVSYS_CH0CTRL=EVSYS_QDIEN_bm|EVSYS_QDEN_bm|EVSYS_DIGFILT_2SAMPLES_gc;
	// sets the timer to use the quadrature decoder event action and defines the clock source
	TCC1_CTRLD=TC_EVACT_QDEC_gc|TC_EVSEL_CH0_gc;
	TCC1_CTRLA=TC_CLKSEL_DIV1_gc; 

	qdec_config_phase_pins(config, &PORTC, 5, false, 2);
	qdec_config_revolution(config, 8192);
	qdec_enabled(config);
}

void initLed(void)
{
	PORTA.DIR |= (1 << 7);
	PORTA.OUT &= (0 << 7);	
}

void initUart(void)
{
	/*
	Initialize the USART
		-> 19200 Baud @ 2 MHz with CLK2X = 0, BSCALE = -5
		-> Rx InterruptW
		-> Use Rx and Tx
		-> 8N1
	*/
	USARTE0.BAUDCTRLA = 0xB0 & 0xFF;
	USARTE0.BAUDCTRLB = ((0xB0 & 0xF00) >> 0x08);
	USARTE0.BAUDCTRLB |= ((-5 & 0x0F) << 0x04);
	USARTE0.CTRLA = USART_RXCINTLVL_LO_gc;
	USARTE0.STATUS |= USART_RXCIF_bm;
	USARTE0.CTRLB = USART_TXEN_bm | USART_RXEN_bm;
	USARTE0.CTRLC = USART_CHSIZE_8BIT_gc;
	USARTE0.CTRLC &= ~(USART_PMODE0_bm | USART_PMODE1_bm | USART_SBMODE_bm);
	PORTE.DIR = 0x08;
	
	while(!(USARTE0.STATUS & USART_DREIF_bm));
	
	PMIC.CTRL = PMIC_LOLVLEN_bm;
	sei();
}

void sendMsg(char* data)
{
	while(*data)
	{
		while(!(USARTE0.STATUS & USART_DREIF_bm));
		USARTE0.DATA = *data++;
	}
}

uint8_t getChar(void)
{	
	// blocking
	while (!(USARTE0.STATUS & USART_RXCIF_bm));
	return ((uint8_t)(&USARTE0)->DATA);
}

void putChar(uint8_t chr)
{
	while(!(USARTE0.STATUS & USART_DREIF_bm));
	USARTE0.DATA = chr;
	while(!(USARTE0.STATUS & USART_DREIF_bm));
	USARTE0.DATA = '\n';
}
